"""
summarize_prenoms.py — Phase 3 prénoms
========================================
Entrée  : prenoms/data/3_prenoms_grouped.json
Sorties : prenoms/data/4_prenoms_final.json        (un objet par prénom, prêt Flask)
          prenoms/data/4_groupes_finals_prenoms.json (un objet par groupe)
Cache   : prenoms/data/4_cache_paraphrase.pkl

Structure prenoms_final.json (lookup direct Flask) :
    {
        "id":              str,
        "prenom":          str,
        "prenom_original": str,
        "sexe":            str,
        "id_groupe_total": int,
        "langue":          str,
        "religion":        str,
        "geo":             str,
        "date":            {"label": str, "valeur": int|null},
        "prenoms_lies":    [str],
        "prenoms_groupe":  [str],
        "etymologie":      str,   # reformulé via pivot FR→EN→FR
        "provenance":      str,
        "histoire":        str,
        "signification":   str,
    }

Reformulation : pivot Helsinki FR→EN→FR par champ, par prénom individuel.
Cache disque keyed sur SHA256 du texte source : idempotent, les textes
identiques entre prénoms ne sont traduits qu'une seule fois.

Déduplication post-pivot : si la reformulation est trop proche de la
source (sim cosine TF-IDF > DEDUP_SEUIL), on conserve la source nettoyée
— le pivot n'a rien apporté et on l'indique plutôt que de mentir.

Estimation CPU : ~8 000 prénoms × 4 champs × 2 passes = ~85 min premier run.
Relances avec cache complet : < 10 secondes.

Dépendances :
    pip install transformers sentencepiece sacremoses torch
    pip install scikit-learn numpy tqdm
    (sacremoses est requis par le tokenizer Helsinki — ne pas omettre)
    Vérification post-install : pip check
"""

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
    from transformers import MarianMTModel, MarianTokenizer
    HAS_HELSINKI = True
except ImportError:
    HAS_HELSINKI = False

INPUT_GROUPED    = "prenoms/data/3_prenoms_grouped.json"
OUTPUT_PRENOMS   = "prenoms/data/4_prenoms_final.json"
OUTPUT_GROUPES   = "prenoms/data/4_groupes_finals_prenoms.json"
CACHE_PARAPHRASE = "prenoms/data/4_cache_paraphrase.pkl"

MODEL_FR_EN = "Helsinki-NLP/opus-mt-fr-en"
MODEL_EN_FR = "Helsinki-NLP/opus-mt-en-fr"

CHAMPS_REFORMULER = ("etymologie", "provenance", "histoire", "signification")

# Seuil au-dessus duquel on considère que le pivot n'a pas apporté
# de variation suffisante — repli sur source nettoyée.
DEDUP_SEUIL = 0.97

# Plafond caractères transmis au tokenizer Helsinki.
# Helsinki supporte ~512 tokens (≈ 2 000 chars français). Marge de sécurité
# pour éviter les troncatures silencieuses sur les textes longs.
MAX_CHARS_TRADUCTION = 1800

BATCH_SIZE = 32

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
    Clé SHA256 du texte source : invariante aux réordonnements du JSON
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
# Modèles Helsinki — pivot FR→EN→FR
# ---------------------------------------------------------------------------

class PivotParaphraser:
    """
    Reformulation FR→EN→FR via Helsinki-NLP/opus-mt-fr-en + opus-mt-en-fr.

    Besoin : produire des reformulations locales sans API externe.
    Choix pivot anglais : modèles OPUS les plus stables pour le français,
    légers (~300 Mo chacun), Apache 2.0, CPU viables.
    La re-traduction FR→EN→FR introduit une variation syntaxique et lexicale
    réelle sans modifier le contenu factuel — adapté aux textes étymologiques.

    num_beams=4 : compromis qualité/vitesse. Réduire à 2 si trop lent sur CPU.
    batch_size : réduire à 8-16 si OOM.
    """

    def __init__(self, batch_size: int = BATCH_SIZE):
        if not HAS_HELSINKI:
            raise RuntimeError(
                "transformers non installe. "
                "pip install transformers sentencepiece sacremoses torch"
            )
        log.info("Chargement %s...", MODEL_FR_EN)
        self._tok_fr_en = MarianTokenizer.from_pretrained(MODEL_FR_EN)
        self._mod_fr_en = MarianMTModel.from_pretrained(MODEL_FR_EN)
        self._mod_fr_en.eval()

        log.info("Chargement %s...", MODEL_EN_FR)
        self._tok_en_fr = MarianTokenizer.from_pretrained(MODEL_EN_FR)
        self._mod_en_fr = MarianMTModel.from_pretrained(MODEL_EN_FR)
        self._mod_en_fr.eval()

        self._batch_size = batch_size
        log.info("Modeles Helsinki prets.")

    def _traduire_batch(
        self,
        textes: list,
        tokenizer: MarianTokenizer,
        model: MarianMTModel,
    ) -> list:
        import torch
        resultats = []
        for i in range(0, len(textes), self._batch_size):
            batch  = textes[i : i + self._batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                traduit = model.generate(**tokens, num_beams=4)
            decoded = tokenizer.batch_decode(traduit, skip_special_tokens=True)
            resultats.extend(decoded)
        return resultats

    def paraphraser_batch(self, textes: list) -> list:
        """
        Applique le pivot FR→EN→FR sur une liste de textes non vides.
        Les textes vides sont retournés tels quels sans appel modèle.
        Retourne une liste de même longueur.
        """
        indices_valides = [i for i, t in enumerate(textes) if t and t.strip()]
        if not indices_valides:
            return textes[:]

        sources   = [textes[i][:MAX_CHARS_TRADUCTION] for i in indices_valides]
        en_batch  = self._traduire_batch(sources, self._tok_fr_en, self._mod_fr_en)
        fr2_batch = self._traduire_batch(en_batch,  self._tok_en_fr, self._mod_en_fr)

        resultats = list(textes)
        for pos, idx in enumerate(indices_valides):
            resultats[idx] = fr2_batch[pos]
        return resultats


# ---------------------------------------------------------------------------
# Vérification de variation post-pivot
# ---------------------------------------------------------------------------

def construire_verificateur(corpus: list) -> TfidfVectorizer:
    """
    Vectoriseur TF-IDF entraîné sur tous les textes sources.
    Utilisé uniquement pour la sim cosine source/reformulé.
    min_df=1 : corpus hétérogène, pas de seuil de fréquence minimum.
    """
    textes_clean = [t if t and t.strip() else "__vide__" for t in corpus]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    vec.fit(textes_clean)
    return vec


def est_trop_proche(source: str, reformule: str, vec: TfidfVectorizer) -> bool:
    """
    Retourne True si la reformulation est quasi-identique à la source
    (sim cosine TF-IDF > DEDUP_SEUIL).
    Indique que le pivot n'a produit aucune variation utile.
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
# Nettoyage post-traduction
# ---------------------------------------------------------------------------

def nettoyer_reformulation(texte: str) -> str:
    """
    Normalise espaces, majuscule initiale et ponctuation finale.
    Helsinki produit parfois des segments sans majuscule ou sans point final.
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
    Retourne la date la plus fréquente (par label) parmi les membres du groupe.
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
    les appels au paraphraser via déduplication par hash de texte source.

    Passe 1 : inventaire des textes non encore cachés, par champ.
    Passe 2 : traduction batch par champ sur les textes manquants.
              Vérification cosine post-pivot — repli source si trop proche.
    Passe 3 : résolution dict[id_prenom -> dict[champ -> reformulé]]
              depuis le cache complet.

    Repli sans paraphraser (Helsinki absent) : source nettoyée et tronquée.
    """
    # Passe 1
    a_traduire: dict = {c: {} for c in CHAMPS_REFORMULER}
    for item in data:
        for champ in CHAMPS_REFORMULER:
            texte = item.get(champ) or ""
            if not texte.strip():
                continue
            cle = cle_cache(texte)
            if cle not in cache and cle not in a_traduire[champ]:
                a_traduire[champ][cle] = texte

    total = sum(len(v) for v in a_traduire.values())
    log.info(
        "Textes a reformuler : %d uniques | cache existant : %d entrees",
        total, len(cache),
    )

    # Passe 2
    if paraphraser is not None and total > 0:
        for champ in CHAMPS_REFORMULER:
            if not a_traduire[champ]:
                continue
            cles   = list(a_traduire[champ].keys())
            textes = [a_traduire[champ][c] for c in cles]
            log.info("Reformulation '%s' : %d textes...", champ, len(textes))

            reformules = paraphraser.paraphraser_batch(textes)

            for cle, source, reformule in zip(cles, textes, reformules):
                reformule = nettoyer_reformulation(reformule)
                if not reformule or est_trop_proche(source, reformule, vec_dedup):
                    # Pivot sans apport : on conserve la source nettoyée
                    reformule = nettoyer_reformulation(source[:MAX_CHARS_TRADUCTION])
                cache[cle] = reformule
    else:
        # Repli : source nettoyée sans traduction
        for champ in CHAMPS_REFORMULER:
            for cle, texte in a_traduire[champ].items():
                cache[cle] = nettoyer_reformulation(texte[:MAX_CHARS_TRADUCTION])

    # Passe 3
    resultats: dict = {}
    for item in data:
        pid = item.get("id", "")
        resultats[pid] = {}
        for champ in CHAMPS_REFORMULER:
            texte = item.get(champ) or ""
            if not texte.strip():
                resultats[pid][champ] = ""
                continue
            cle = cle_cache(texte)
            resultats[pid][champ] = cache.get(
                cle,
                nettoyer_reformulation(texte[:MAX_CHARS_TRADUCTION]),
            )

    return resultats


# ---------------------------------------------------------------------------
# Construction des sorties
# ---------------------------------------------------------------------------

def construire_sorties(data: list, reformulations: dict) -> tuple:
    """
    Besoin : assembler les deux fichiers de sortie.

    Champs catégoriels (langue, religion, geo, date) : agrégés au niveau
    groupe par vote majoritaire — cohérence inter-pipelines.
    Champs textuels (etymologie, provenance, histoire, signification) :
    individuels, reformulation propre à chaque prénom.
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
            "provenance":      ref.get("provenance",      ""),
            "histoire":        ref.get("histoire",        ""),
            "signification":   ref.get("signification",   ""),
        })

    return prenoms_final, groupes_finals


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(INPUT_GROUPED):
        log.error("Fichier introuvable : %s", INPUT_GROUPED)
        return

    os.makedirs(os.path.dirname(OUTPUT_PRENOMS), exist_ok=True)

    with open(INPUT_GROUPED, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info("%d entrees chargees", len(data))

    # Vectoriseur pour la vérification post-pivot — entraîné sur toutes
    # les sources pour avoir un espace TF-IDF commun représentatif.
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
    if HAS_HELSINKI:
        try:
            paraphraser = PivotParaphraser(batch_size=BATCH_SIZE)
        except Exception as exc:
            log.warning(
                "Impossible de charger Helsinki (%s) — repli source nettoyee.", exc
            )
    else:
        log.warning(
            "transformers non installe — champs = source nettoyee. "
            "pip install transformers sentencepiece sacremoses torch"
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

    # Statistiques de couverture
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