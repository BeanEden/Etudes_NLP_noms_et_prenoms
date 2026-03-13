"""
summarize_prenoms.py — Phase 3 prénoms
========================================
Entrée  : prenoms/data/3_prenoms_grouped.json
Sorties : prenoms/data/4_prenoms_final.json        (un objet par prénom, prêt Flask)
          prenoms/data/4_groupes_finals_prenoms.json (un objet par groupe)
Cache   : prenoms/data/4_cache_paraphrase.pkl

Reformulation : CamemBERT2CamemBERT (EncoderDecoderModel) fine-tuné sur MLSUM FR.
    Modèle  : mrm8488/camembert2camembert_shared-finetuned-french-summarization
    API     : RobertaTokenizerFast + EncoderDecoderModel (pas MarianMT)
    Poids   : 559 Mo (safetensors), CPU viable
    Objectif: diversification du texte affiché — éviter le copié-collé visible

Cache disque keyed sur SHA256 du texte source : idempotent, les textes
identiques entre prénoms ne sont traduits qu'une seule fois.

Gestion des textes longs (> MAX_TOKENS) :
    Segmentation par phrase, reformulation par blocs de MAX_TOKENS tokens,
    concaténation des sorties. Helsinki tronquait silencieusement — ici on gère.

Vérification post-reformulation :
    Si sim cosine TF-IDF source/reformulé > DEDUP_SEUIL, le modèle n'a rien
    apporté (texte trop court ou trop générique) — repli sur source nettoyée.

Estimation CPU :
    ~5 000 textes uniques × ~1.5 s/texte = ~2 h premier run avec num_beams=4.
    Réduire num_beams à 1 (greedy) pour passer sous 20 min.
    NUM_BEAMS est configurable via --num_beams.
    Relances avec cache complet : < 10 secondes.

Dépendances :
    pip install transformers sentencepiece torch
    pip install scikit-learn numpy tqdm
    Vérification post-install : pip check
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    import torch
    from transformers import RobertaTokenizerFast, EncoderDecoderModel
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

INPUT_GROUPED    = "prenoms/data/3_prenoms_grouped.json"
OUTPUT_PRENOMS   = "prenoms/data/4_prenoms_final.json"
OUTPUT_GROUPES   = "prenoms/data/4_groupes_finals_prenoms.json"
CACHE_PARAPHRASE = "prenoms/data/4_cache_paraphrase.pkl"

MODEL_CKPT = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"

CHAMPS_REFORMULER = ("etymologie", "provenance", "histoire", "signification")

# Seuil au-dessus duquel la reformulation est jugée trop proche de la source.
# Repli sur source nettoyée si sim cosine TF-IDF > DEDUP_SEUIL.
DEDUP_SEUIL = 0.97

# Nombre de tokens maximum transmis au modèle en une passe.
# EncoderDecoderModel / CamemBERT supporte 512 tokens.
# Marge à 480 pour laisser de l'espace aux tokens spéciaux.
MAX_TOKENS = 480

# num_beams=1 (greedy) : ~20 min sur CPU pour 8 000 prénoms.
# num_beams=4 (beam search) : meilleure qualité, ~2 h sur CPU.
NUM_BEAMS = 1

# Longueur minimale de sortie générée (tokens). Evite les outputs trop courts
# sur les textes riches en information factuelle.
MIN_LENGTH_OUT = 20

# Taille du batch pour le generate(). Réduire à 4 si OOM sur CPU.
BATCH_SIZE = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache disque
# ---------------------------------------------------------------------------

def charger_cache(path: str) -> dict:
    """
    Cache dict[sha256_hex -> texte_reformule].
    Clé SHA256 du texte source : invariante aux réordonnements JSON
    et aux relances partielles.
    """
    if os.path.isfile(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)
        log.info("Cache paraphrase charge : %d entrees (%s)", len(cache), path)
        return cache
    log.info("Pas de cache — premier run.")
    return {}


def sauvegarder_cache(path: str, cache: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    log.info("Cache paraphrase sauvegarde : %d entrees", len(cache))


def cle_cache(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Modèle CamemBERT2CamemBERT
# ---------------------------------------------------------------------------

class CamembertParaphraser:
    """
    Reformulation française via EncoderDecoderModel (CamemBERT2CamemBERT).

    Besoin : diversification du texte affiché sans API externe, en français natif.

    Choix vs Helsinki FR→EN→FR :
        - Natif français : pas de double erreur de traduction sur les noms propres
          et termes étymologiques.
        - Fine-tuné sur MLSUM FR (1.5M paires article/résumé) : reformule en
          conservant les faits, varie la structure syntaxique.
        - 559 Mo vs ~600 Mo par modèle Helsinki (2 modèles = 1.2 Go).
        - API : RobertaTokenizerFast + EncoderDecoderModel (pas MarianMT).

    Gestion des textes longs :
        Segmentation par phrase puis encodage par blocs de MAX_TOKENS tokens.
        Chaque bloc est reformulé indépendamment, les sorties sont concaténées.
        Evite la troncature silencieuse du tokenizer.

    num_beams=1 (greedy) par défaut : ~20 min sur CPU pour 8 000 prénoms.
    """

    def __init__(self, num_beams: int = NUM_BEAMS, batch_size: int = BATCH_SIZE):
        if not HAS_MODEL:
            raise RuntimeError(
                "transformers ou torch non installe. "
                "pip install transformers sentencepiece torch"
            )
        log.info("Chargement %s...", MODEL_CKPT)
        self._tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_CKPT)
        self._model     = EncoderDecoderModel.from_pretrained(MODEL_CKPT)
        self._model.eval()
        self._num_beams  = num_beams
        self._batch_size = batch_size
        self._device     = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        log.info(
            "Modele pret (device=%s, num_beams=%d, batch_size=%d).",
            self._device, num_beams, batch_size,
        )

    def _segmenter(self, texte: str) -> list:
        """
        Découpe un texte en phrases via regex.
        Retourne une liste de blocs dont chacun tient en MAX_TOKENS tokens.
        Les phrases sont accumulées dans un bloc jusqu'au dépassement du seuil,
        puis un nouveau bloc est ouvert — jamais de coupure en milieu de phrase.
        """
        phrases  = re.split(r"(?<=[.!?])\s+", texte.strip())
        blocs    = []
        courant  = []
        n_tokens = 0

        for phrase in phrases:
            t = len(self._tokenizer.encode(phrase, add_special_tokens=False))
            if n_tokens + t > MAX_TOKENS and courant:
                blocs.append(" ".join(courant))
                courant  = [phrase]
                n_tokens = t
            else:
                courant.append(phrase)
                n_tokens += t

        if courant:
            blocs.append(" ".join(courant))

        return blocs if blocs else [texte]

    def _reformuler_blocs(self, blocs: list) -> str:
        """
        Reformule une liste de blocs via generate() et concatène les sorties.
        padding="max_length" obligatoire pour EncoderDecoderModel (comportement
        dégradé avec padding dynamique sur ce checkpoint).
        """
        resultats = []
        for i in range(0, len(blocs), self._batch_size):
            batch  = blocs[i : i + self._batch_size]
            inputs = self._tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            input_ids      = inputs.input_ids.to(self._device)
            attention_mask = inputs.attention_mask.to(self._device)

            with torch.no_grad():
                output = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    num_beams=self._num_beams,
                    min_length=MIN_LENGTH_OUT,
                    no_repeat_ngram_size=3,
                )
            decoded = self._tokenizer.batch_decode(output, skip_special_tokens=True)
            resultats.extend(decoded)

        return " ".join(resultats)

    def reformuler(self, texte: str) -> str:
        """
        Point d'entrée principal : segmente si nécessaire, reformule, retourne.
        Les textes vides sont retournés tels quels sans appel modèle.
        """
        if not texte or not texte.strip():
            return texte
        blocs = self._segmenter(texte)
        return self._reformuler_blocs(blocs)

    def reformuler_batch(self, textes: list) -> list:
        """
        Reformule une liste de textes en préservant l'ordre et les vides.
        Regroupe les blocs de tous les textes en un seul passage batch
        pour maximiser l'utilisation CPU/GPU.

        Stratégie :
            1. Segmente chaque texte en blocs.
            2. Aplatit tous les blocs dans une liste unique avec index de retour.
            3. Appelle generate() en batch sur la liste aplatie.
            4. Réassemble les sorties par texte source.
        """
        indices_valides = [i for i, t in enumerate(textes) if t and t.strip()]
        if not indices_valides:
            return list(textes)

        # Segmentation — blocs_map[i] = liste de blocs du texte i
        blocs_map     = {i: self._segmenter(textes[i]) for i in indices_valides}
        tous_blocs    = []
        retour_index  = []   # (idx_texte, idx_bloc_dans_texte)
        for i, blocs in blocs_map.items():
            for b_idx, bloc in enumerate(blocs):
                tous_blocs.append(bloc)
                retour_index.append((i, b_idx))

        # Generate en batch unique
        sorties_blocs = []
        for k in range(0, len(tous_blocs), self._batch_size):
            batch  = tous_blocs[k : k + self._batch_size]
            inputs = self._tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            input_ids      = inputs.input_ids.to(self._device)
            attention_mask = inputs.attention_mask.to(self._device)
            with torch.no_grad():
                output = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    num_beams=self._num_beams,
                    min_length=MIN_LENGTH_OUT,
                    no_repeat_ngram_size=3,
                )
            decoded = self._tokenizer.batch_decode(output, skip_special_tokens=True)
            sorties_blocs.extend(decoded)

        # Réassemblage par texte
        texte_blocs_sortie: dict = defaultdict(list)
        for sortie, (i, b_idx) in zip(sorties_blocs, retour_index):
            texte_blocs_sortie[i].append((b_idx, sortie))

        resultats = list(textes)
        for i in indices_valides:
            blocs_sorted = sorted(texte_blocs_sortie[i], key=lambda x: x[0])
            resultats[i] = " ".join(s for _, s in blocs_sorted)

        return resultats


# ---------------------------------------------------------------------------
# Vérification de variation post-reformulation
# ---------------------------------------------------------------------------

def construire_verificateur(corpus: list) -> TfidfVectorizer:
    """
    Vectoriseur TF-IDF entraîné sur tous les textes sources.
    Utilisé uniquement pour la sim cosine source/reformulé.
    """
    textes_clean = [t if t and t.strip() else "__vide__" for t in corpus]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    vec.fit(textes_clean)
    return vec


def est_trop_proche(source: str, reformule: str, vec: TfidfVectorizer) -> bool:
    """
    True si la reformulation est quasi-identique à la source (sim > DEDUP_SEUIL).
    Indique que le modèle n'a produit aucune variation utile.
    """
    if not source or not reformule:
        return False
    try:
        vecs = vec.transform([source, reformule])
        sim  = float(cosine_similarity(vecs[0], vecs[1])[0, 0])
        return sim > DEDUP_SEUIL
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Nettoyage post-reformulation
# ---------------------------------------------------------------------------

def nettoyer_reformulation(texte: str) -> str:
    """
    Normalise espaces, majuscule initiale et ponctuation finale.
    Le modèle CamemBERT2CamemBERT produit parfois des segments sans
    majuscule initiale ou avec des espaces multiples.
    """
    texte = re.sub(r"\s+", " ", texte).strip()
    if not texte:
        return ""
    texte = texte[0].upper() + texte[1:]
    if texte[-1] not in ".!?":
        texte += "."
    return texte


# ---------------------------------------------------------------------------
# Agrégation catégorielle au niveau groupe
# ---------------------------------------------------------------------------

def valeur_majoritaire(valeurs: list) -> str:
    non_vides = [v for v in valeurs if v]
    if not non_vides:
        return ""
    return Counter(non_vides).most_common(1)[0][0]


def date_majoritaire(dates: list) -> dict:
    """
    Retourne la date la plus fréquente (par label) parmi les membres.
    En cas d'égalité, préfère la date la plus ancienne.
    """
    valides = [d for d in dates if d and d.get("valeur")]
    if not valides:
        return {"label": "", "valeur": None}
    counter = Counter(d["label"] for d in valides)
    label   = counter.most_common(1)[0][0]
    valeur  = next(d["valeur"] for d in valides if d["label"] == label)
    return {"label": label, "valeur": valeur}


# ---------------------------------------------------------------------------
# Pipeline de reformulation avec cache
# ---------------------------------------------------------------------------

def reformuler_corpus(
    data: list,
    paraphraser,
    cache: dict,
    vec_dedup: TfidfVectorizer,
) -> dict:
    """
    Besoin : reformuler chaque champ de chaque prénom en minimisant
    les appels au modèle via déduplication par hash de texte source.

    Passe 1 : inventaire des textes non encore cachés, par champ.
    Passe 2 : reformulation batch par champ sur les textes manquants.
              Vérification cosine post-reformulation — repli source si trop proche.
    Passe 3 : résolution dict[id_prenom -> dict[champ -> reformulé]].

    Repli sans modèle : source nettoyée sans reformulation.
    """
    # Passe 1
    a_reformuler: dict = {c: {} for c in CHAMPS_REFORMULER}
    for item in data:
        for champ in CHAMPS_REFORMULER:
            texte = item.get(champ) or ""
            if not texte.strip():
                continue
            cle = cle_cache(texte)
            if cle not in cache and cle not in a_reformuler[champ]:
                a_reformuler[champ][cle] = texte

    total = sum(len(v) for v in a_reformuler.values())
    log.info(
        "Textes a reformuler : %d uniques | cache existant : %d entrees",
        total, len(cache),
    )

    # Passe 2
    if paraphraser is not None and total > 0:
        for champ in CHAMPS_REFORMULER:
            if not a_reformuler[champ]:
                continue
            cles   = list(a_reformuler[champ].keys())
            textes = [a_reformuler[champ][c] for c in cles]
            log.info("Reformulation '%s' : %d textes...", champ, len(textes))

            reformules = paraphraser.reformuler_batch(textes)

            n_repli = 0
            for cle, source, reformule in zip(cles, textes, reformules):
                reformule = nettoyer_reformulation(reformule)
                if not reformule or est_trop_proche(source, reformule, vec_dedup):
                    reformule = nettoyer_reformulation(source)
                    n_repli  += 1
                cache[cle] = reformule

            log.info(
                "  -> %d/%d replis sur source (reformulation sans apport)",
                n_repli, len(cles),
            )
    else:
        # Repli : source nettoyée sans reformulation
        for champ in CHAMPS_REFORMULER:
            for cle, texte in a_reformuler[champ].items():
                cache[cle] = nettoyer_reformulation(texte)

    # Passe 3
    resultats: dict = {}
    for item in data:
        pid = item.get("id", "")
        resultats[pid] = {}
        for champ in CHAMPS_REFORMULER:
            texte = item.get(champ) or ""
            if not texte.strip():
                resultats[pid][champ]          = ""
                resultats[pid][champ + "_v"]   = ""
                continue
            cle = cle_cache(texte)
            source_nettoyee = nettoyer_reformulation(texte)
            reformule       = cache.get(cle, source_nettoyee)
            resultats[pid][champ]          = source_nettoyee   # texte brut nettoyé
            resultats[pid][champ + "_v"]   = reformule          # version CamemBERT

    return resultats


# ---------------------------------------------------------------------------
# Construction des sorties
# ---------------------------------------------------------------------------

def construire_sorties(data: list, reformulations: dict) -> tuple:
    """
    Assemble les deux fichiers de sortie.

    Champs catégoriels (langue, religion, geo, date) : agrégés par vote
    majoritaire au niveau groupe.
    Champs textuels : individuels, reformulation propre à chaque prénom.
    prenoms_lies : union dédupliquée sur tous les membres du groupe.
    """
    groupes: dict = defaultdict(list)
    for item in data:
        groupes[item["id_groupe_total"]].append(item)

    groupes_finals = []
    meta_groupe    = {}

    log.info("Aggregation de %d groupes...", len(groupes))

    for gid, membres in tqdm(sorted(groupes.items()), desc="Groupes", unit="groupe"):
        langue   = valeur_majoritaire([m.get("langue",   "") for m in membres])
        religion = valeur_majoritaire([m.get("religion", "") for m in membres])
        geo      = valeur_majoritaire([m.get("geo",      "") for m in membres])
        date     = date_majoritaire([m.get("date", {})       for m in membres])

        prenoms_lies_groupe = list(dict.fromkeys(
            p for m in membres for p in m.get("prenoms_lies", [])
        ))
        prenoms_du_groupe = [m["prenom"] for m in membres]

        meta_groupe[gid] = {
            "langue":         langue,
            "religion":       religion,
            "geo":            geo,
            "date":           date,
            "prenoms_lies":   prenoms_lies_groupe,
            "prenoms_groupe": prenoms_du_groupe,
        }

        groupes_finals.append({
            "id_groupe_total": gid,
            "prenoms":         sorted(prenoms_du_groupe),
            "langue":          langue,
            "religion":        religion,
            "geo":             geo,
            "date":            date,
            "prenoms_lies":    prenoms_lies_groupe,
        })

    log.info("Construction de %d entrees prenoms_final...", len(data))
    prenoms_final = []
    for item in tqdm(data, desc="Prenoms", unit="prenom"):
        pid  = item.get("id", "")
        gid  = item["id_groupe_total"]
        meta = meta_groupe.get(gid, {})
        ref  = reformulations.get(pid, {})

        prenoms_final.append({
            "id":              pid,
            "prenom":          item.get("prenom", ""),
            "prenom_original": item.get("prenom_original", ""),
            "sexe":            item.get("sexe", ""),
            "id_groupe_total": gid,
            "langue":          meta.get("langue",         ""),
            "religion":        meta.get("religion",       ""),
            "geo":             meta.get("geo",             ""),
            "date":            meta.get("date",            {"label": "", "valeur": None}),
            "prenoms_lies":    meta.get("prenoms_lies",   []),
            "prenoms_groupe":  meta.get("prenoms_groupe", []),
            "etymologie":      ref.get("etymologie",      ""),
            "etymologie_v":    ref.get("etymologie_v",    ""),
            "provenance":      ref.get("provenance",      ""),
            "provenance_v":    ref.get("provenance_v",    ""),
            "histoire":        ref.get("histoire",        ""),
            "histoire_v":      ref.get("histoire_v",      ""),
            "signification":   ref.get("signification",   ""),
            "signification_v": ref.get("signification_v", ""),
        })

    return prenoms_final, groupes_finals


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    global NUM_BEAMS, BATCH_SIZE, DEDUP_SEUIL

    parser = argparse.ArgumentParser(
        description="Phase 3 prénoms — reformulation CamemBERT2CamemBERT."
    )
    parser.add_argument(
        "--num_beams", type=int, default=NUM_BEAMS,
        help="Nombre de beams (1=greedy ~20 min CPU, 4=qualité ~2h CPU).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Taille du batch generate(). Réduire si OOM.",
    )
    parser.add_argument(
        "--dedup_seuil", type=float, default=DEDUP_SEUIL,
        help="Seuil cosine TF-IDF pour repli sur source (défaut 0.97).",
    )
    args = parser.parse_args()

    NUM_BEAMS   = args.num_beams
    BATCH_SIZE  = args.batch_size
    DEDUP_SEUIL = args.dedup_seuil

    if not os.path.isfile(INPUT_GROUPED):
        log.error("Fichier introuvable : %s", INPUT_GROUPED)
        return

    os.makedirs(os.path.dirname(OUTPUT_PRENOMS), exist_ok=True)

    with open(INPUT_GROUPED, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info("%d entrees chargees", len(data))

    tous_textes_sources = [
        t
        for item in data
        for champ in CHAMPS_REFORMULER
        if (t := (item.get(champ) or "")) and t.strip()
    ]
    log.info(
        "Construction vectoriseur deduplication (%d textes sources)...",
        len(tous_textes_sources),
    )
    vec_dedup = construire_verificateur(tous_textes_sources)

    cache = charger_cache(CACHE_PARAPHRASE)

    paraphraser = None
    if HAS_MODEL:
        try:
            paraphraser = CamembertParaphraser(
                num_beams=NUM_BEAMS,
                batch_size=BATCH_SIZE,
            )
        except Exception as exc:
            log.warning(
                "Impossible de charger le modele (%s) — repli source nettoyee.", exc
            )
    else:
        log.warning(
            "transformers/torch non installe — champs = source nettoyee. "
            "pip install transformers sentencepiece torch"
        )

    reformulations = reformuler_corpus(data, paraphraser, cache, vec_dedup)
    sauvegarder_cache(CACHE_PARAPHRASE, cache)

    prenoms_final, groupes_finals = construire_sorties(data, reformulations)

    with open(OUTPUT_PRENOMS, "w", encoding="utf-8") as f:
        json.dump(prenoms_final, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d prenoms)", OUTPUT_PRENOMS, len(prenoms_final))

    with open(OUTPUT_GROUPES, "w", encoding="utf-8") as f:
        json.dump(groupes_finals, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d groupes)", OUTPUT_GROUPES, len(groupes_finals))

    n_groupes = len(groupes_finals)
    for champ in ("langue", "religion", "geo"):
        nb = sum(1 for g in groupes_finals if g[champ])
        log.info("Couverture groupe %s : %.1f%%", champ, 100 * nb / n_groupes)

    n_total = len(data) * len(CHAMPS_REFORMULER)
    n_vides = sum(
        1
        for item in prenoms_final
        for champ in CHAMPS_REFORMULER
        if not item.get(champ)
    )
    log.info(
        "Couverture reformulation : %d/%d champs renseignes (%.1f%%)",
        n_total - n_vides, n_total,
        100 * (n_total - n_vides) / max(n_total, 1),
    )


if __name__ == "__main__":
    main()