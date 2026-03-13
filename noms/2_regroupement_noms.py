"""
Clustering des noms — Levenshtein + TF-IDF sparse sur texte brut
=================================================================
Entrée  : ./data/noms_clean.json   (produit par prepare_noms.py)
Sorties : ./data/noms_grouped.json
          ./data/groupes_noms.json
          ./data/evaluation_sample.json
Cache   : ./data/paires_scores.pkl

Pipeline :
    1. Présélection par préfixe commun -> candidats
    2. JaroWinkler >= SEUIL_LEV_PRESELECTION -> filtre orthographique
    3. Pour chaque paire retenue :
       a. sim_tfidf   = cosine TF-IDF(texte_brut_a, texte_brut_b)
       b. bonus_langue = 1.0 si même langue source détectée
       c. bonus_geo    = 1.0 si même provenance géo détectée
       d. score_ctx    = 0.5*sim_tfidf + 0.3*b_langue + 0.2*b_geo
       e. t = min(len_min, LONGUEUR_MAX) / LONGUEUR_MAX
       f. score_final  = t * sim_lev + (1-t) * score_ctx
    4. Union-Find : groupes prepare_noms injectés + nouvelles paires

Pourquoi TF-IDF plutôt que BERT :
    Les textes d'origine sont des sacs de mots-clés courts (5-15 tokens).
    BERT est optimisé pour la syntaxe de phrases — peu adapté ici.
    TF-IDF cosine est exact, instantané et plus fiable sur ce type de corpus.

Installation :
    pip install rapidfuzz scikit-learn numpy tqdm
"""

import json
import logging
import os
import pickle
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_JSON             = "noms/data/1_noms_clean.json"
OUTPUT_GROUPED         = "noms/data/2_noms_grouped.json"
OUTPUT_GROUPES         = "noms/data/2_groupes_noms.json"
OUTPUT_EVAL            = "noms/data/2_evaluation_sample.json"
CACHE_PAIRES           = "noms/data/2_paires_scores.pkl"

SEUIL_LEV_PRESELECTION = 0.75   # JaroWinkler minimal pour présélection
SEUIL_SCORE_FINAL      = 0.55   # score pondéré final minimal
LONGUEUR_MAX           = 15     # plafond pondération longueur
PREFIXE_LEN            = 2      # longueur préfixe pour le blocage
LEV_K                  = 40     # candidats max par nom
N_EVAL_PAIRES          = 200
NB_WORKERS             = os.cpu_count()-2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Listes fermées : langue source + provenance géographique
# ---------------------------------------------------------------------------

LANGUES_SOURCE = {
    "arabe", "hébreux", "hébreu", "latin", "germanique", "germain",
    "celtique", "celte", "breton", "bretonne", "gaulois", "gauloise",
    "grec", "grecque", "scandinave", "norse", "viking", "anglais",
    "occitan", "provençal", "flamand", "alsacien", "basque", "catalan",
    "espagnol", "portugais", "italien", "slave", "sémitique", "berbère",
    "normand", "normande", "francique", "mérovingien",
    "vieux français", "vieux haut allemand",
}

REGIONS_GEO = {
    "bretagne", "normandie", "alsace", "lorraine", "provence", "languedoc",
    "gascogne", "bourgogne", "auvergne", "poitou", "anjou", "touraine",
    "champagne", "picardie", "flandre", "pays basque", "catalogne",
    "france", "allemagne", "angleterre", "italie", "espagne", "portugal",
    "belgique", "suisse", "jura", "savoie", "dauphiné", "franche comté",
    "midi", "île de france",
}


def extraire_langue(texte: str) -> str:
    if not texte:
        return ""
    mots = set(re.findall(r"\b\w+\b", texte.lower()))
    for langue in LANGUES_SOURCE:
        if all(m in mots for m in langue.split()):
            return langue
    return ""


def extraire_geo(texte: str) -> str:
    if not texte:
        return ""
    mots = set(re.findall(r"\b\w+\b", texte.lower()))
    for region in REGIONS_GEO:
        if all(m in mots for m in region.split()):
            return region
    return ""


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def construire_tfidf(textes_bruts: list) -> tuple:
    """
    Construit la matrice TF-IDF sur les textes bruts dédupliqués.
    Retourne (matrice_dense_float32, texte_to_idx).

    Dense car les workers ProcessPool ne peuvent pas recevoir de matrice sparse
    (non picklable efficacement). La matrice dédupliquée reste petite :
    ~18k textes uniques × ~5k features = ~360 Mo max, souvent bien moins.
    """
    textes_clean   = [t if t and t.strip() else "__vide__" for t in textes_bruts]
    textes_uniques = list(dict.fromkeys(textes_clean))
    texte_to_idx   = {t: i for i, t in enumerate(textes_uniques)}

    log.info(
        "TF-IDF : %d textes uniques / %d total (%.1f%% doublons)",
        len(textes_uniques), len(textes_bruts),
        100 * (1 - len(textes_uniques) / max(len(textes_bruts), 1)),
    )

    vec     = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrice = vec.fit_transform(textes_uniques).toarray().astype(np.float32)
    log.info("Matrice TF-IDF : %s", matrice.shape)
    return matrice, texte_to_idx


# ---------------------------------------------------------------------------
# Présélection par préfixe
# ---------------------------------------------------------------------------

def construire_index_prefixe(noms: list) -> dict:
    idx = defaultdict(list)
    for i, nom in enumerate(noms):
        for longueur in range(PREFIXE_LEN, min(PREFIXE_LEN + 2, len(nom) + 1)):
            idx[nom[:longueur]].append(i)
    return idx


# ---------------------------------------------------------------------------
# Scoring par chunk
# ---------------------------------------------------------------------------

def _scorer_chunk(args: tuple) -> list:
    """
    Worker ProcessPoolExecutor.

    Séquence par nom i :
        1. Candidats préfixe
        2. JaroWinkler >= seuil_lev_pre
        3. score_ctx  = 0.5*tfidf_cosine + 0.3*bonus_langue + 0.2*bonus_geo
        4. score_final = t*sim_lev + (1-t)*score_ctx >= seuil_final
    """
    (noms, id_groupes, textes_bruts, langues, geos,
     tfidf_dense, texte_to_idx,
     idx_prefixe, paires_connues,
     i_start, i_end,
     seuil_lev_pre, seuil_final, longueur_max, lev_k) = args

    from rapidfuzz.distance import JaroWinkler as _JW

    paires = []
    vus    = set()

    for i in range(i_start, i_end):
        nom_i    = noms[i]
        groupe_i = id_groupes[i]
        t_i      = min(len(nom_i), longueur_max) / longueur_max
        langue_i = langues[i]
        geo_i    = geos[i]
        idx_i    = texte_to_idx.get(textes_bruts[i] if textes_bruts[i] else "__vide__", 0)
        emb_i    = tfidf_dense[idx_i]

        cands = set()
        for longueur in range(PREFIXE_LEN, min(PREFIXE_LEN + 2, len(nom_i) + 1)):
            cands.update(idx_prefixe.get(nom_i[:longueur], []))
        cands.discard(i)

        scored = []
        for j in cands:
            if id_groupes[j] == groupe_i:
                continue
            cle = (min(i, j), max(i, j))
            if cle in vus or cle in paires_connues:
                continue
            sim = _JW.similarity(nom_i, noms[j])
            if sim >= seuil_lev_pre * 100:
                scored.append((j, sim))

        scored.sort(key=lambda x: -x[1])

        for j, sim_lev_raw in scored[:lev_k]:
            cle = (min(i, j), max(i, j))
            if cle in vus:
                continue
            vus.add(cle)

            nom_j   = noms[j]
            t       = min(t_i, min(len(nom_j), longueur_max) / longueur_max)
            sim_lev = sim_lev_raw / 100.0

            idx_j     = texte_to_idx.get(textes_bruts[j] if textes_bruts[j] else "__vide__", 0)
            emb_j     = tfidf_dense[idx_j]
            norm_i    = np.linalg.norm(emb_i)
            norm_j    = np.linalg.norm(emb_j)
            sim_tfidf = float(np.dot(emb_i, emb_j) / (norm_i * norm_j + 1e-10))

            b_langue  = 1.0 if langue_i and langue_i == langues[j] else 0.0
            b_geo     = 1.0 if geo_i    and geo_i    == geos[j]    else 0.0
            score_ctx = 0.5 * sim_tfidf + 0.3 * b_langue + 0.2 * b_geo

            sf = t * sim_lev + (1 - t) * score_ctx
            if sf >= seuil_final:
                paires.append((i, j, sf))

    return paires


# ---------------------------------------------------------------------------
# Construction des paires
# ---------------------------------------------------------------------------

def construire_paires(
    noms, id_groupes, textes_bruts, langues, geos,
    tfidf_dense, texte_to_idx, paires_connues,
) -> list:
    idx_prefixe = construire_index_prefixe(noms)
    n           = len(noms)
    nb_workers  = NB_WORKERS or os.cpu_count() or 1
    chunk_size  = max(1, n // nb_workers)
    chunks      = [
        (i_start, min(i_start + chunk_size, n))
        for i_start in range(0, n, chunk_size)
    ]

    log.info(
        "Scoring : %d chunks × ~%d noms | %d workers",
        len(chunks), chunk_size, nb_workers,
    )

    args_list = [
        (noms, id_groupes, textes_bruts, langues, geos,
         tfidf_dense, texte_to_idx,
         dict(idx_prefixe), paires_connues,
         i_start, i_end,
         SEUIL_LEV_PRESELECTION, SEUIL_SCORE_FINAL, LONGUEUR_MAX, LEV_K)
        for i_start, i_end in chunks
    ]

    toutes_paires = []
    vus_global    = set()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=nb_workers) as executor:
        resultats = list(tqdm(
            executor.map(_scorer_chunk, args_list),
            total=len(chunks),
            desc="Scoring chunks",
            unit="chunk",
        ))

    for chunk_result in resultats:
        for i, j, sf in chunk_result:
            cle = (min(i, j), max(i, j))
            if cle not in vus_global:
                toutes_paires.append((i, j, sf))
                vus_global.add(cle)

    log.info("%d nouvelles paires retenues avant filtre NER", len(toutes_paires))
    return toutes_paires


# ---------------------------------------------------------------------------
# Filtre NER
# ---------------------------------------------------------------------------

def filtrer_paires_ner(paires: list, noms: list) -> list:
    """
    Filtre les nouvelles paires en testant si les deux noms sont des 
    entités nommées (PER/LOC/ORG) dans un contexte minimal, 
    pour éliminer les noms communs capturés par erreur.
    """
    if not paires:
        return []
    if not HAS_TRANSFORMERS:
        log.warning("Bibliothèque 'transformers' non trouvée. Pas de filtre NER.")
        return paires

    try:
        log.info("Chargement du modèle NER (Jean-Baptiste/camembert-ner)...")
        ner = pipeline("ner", model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple", device=-1)
    except Exception as e:
        log.warning(f"Erreur chargement NER ({e}). Pas de filtre NER.")
        return paires

    # On extrait la liste unique d'indices de noms impliqués dans les paires
    indices_impliques = set()
    for i, j, _ in paires:
        indices_impliques.add(i)
        indices_impliques.add(j)

    indices_list = list(indices_impliques)
    # On met le nom dans un contexte neutre pour forcer le NER à évaluer sa nature
    textes = [f"Voici {noms[idx].capitalize()}." for idx in indices_list]

    log.info(f"Évaluation NER sur {len(textes)} noms uniques issus des paires...")
    validite_ner = {}
    batch_size   = 64

    for k in tqdm(range(0, len(textes), batch_size), desc="Filtre NER", unit="batch"):
        batch_textes  = textes[k:k+batch_size]
        batch_indices = indices_list[k:k+batch_size]
        try:
            results = ner(batch_textes)
            # pipeline('ner') sur une liste de str retourne une liste de listes de dicts
            for m, res in enumerate(results):
                idx = batch_indices[m]
                # On valide s'il trouve au moins une entité (PER, LOC, ORG...)
                # Jean-Baptiste/camembert-ner renvoie PER, LOC, ORG, MISC.
                est_valide = any(ent.get("entity_group") in ("PER", "LOC", "ORG") for ent in res)
                validite_ner[idx] = est_valide
        except Exception as e:
            for idx in batch_indices:
                validite_ner[idx] = True  # En cas d'erreur de batch, on ne rejette pas

    paires_filtrees = []
    for i, j, sf in paires:
        if validite_ner.get(i, True) and validite_ner.get(j, True):
            paires_filtrees.append((i, j, sf))

    log.info("Filtrage NER : %d -> %d paires.", len(paires), len(paires_filtrees))
    return paires_filtrees


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def labels(self):
        roots   = {self.find(i) for i in range(len(self.parent))}
        root_id = {r: idx + 1 for idx, r in enumerate(sorted(roots))}
        return np.array([root_id[self.find(i)] for i in range(len(self.parent))])


def clusterer(noms, paires, id_groupes_prep):
    n  = len(noms)
    uf = UnionFind(n)

    groupe_to_indices = defaultdict(list)
    for i, gid in enumerate(id_groupes_prep):
        groupe_to_indices[gid].append(i)

    n_injectes = 0
    for indices in tqdm(groupe_to_indices.values(), desc="Injection groupes", unit="groupe"):
        if len(indices) > 1:
            for k in range(1, len(indices)):
                uf.union(indices[0], indices[k])
            n_injectes += 1

    n_avant = len({uf.find(i) for i in range(n)})
    log.info("Après injection : %d composantes (%d groupes)", n_avant, n_injectes)

    for i, j, _ in tqdm(paires, desc="Fusion nouvelles paires", unit="paire"):
        uf.union(i, j)

    labels  = uf.labels()
    n_apres = len(set(labels.tolist()))
    log.info("Après fusion : %d composantes (delta -%d)", n_apres, n_avant - n_apres)
    return labels


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def charger_cache(path):
    if not os.path.isfile(path):
        log.info("Pas de cache — premier run.")
        return set()
    with open(path, "rb") as f:
        paires = pickle.load(f)
    log.info("Cache : %d paires connues", len(paires))
    return paires


def sauvegarder_cache(path, connues, nouvelles):
    toutes = connues | {(min(i, j), max(i, j)) for i, j, _ in nouvelles}
    with open(path, "wb") as f:
        pickle.dump(toutes, f)
    log.info("Cache sauvegardé : %d paires", len(toutes))


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def generer_evaluation(data, labels, n, path):
    """
    Génère n paires 50/50 liées/non-liées pour validation manuelle.
    Inclut langue et géo détectées pour faciliter la validation.
    """
    noms_list  = [d.get("nom", "")     for d in data]
    lang_list  = [d.get("_langue", "") for d in data]
    geo_list   = [d.get("_geo", "")    for d in data]
    label_list = labels.tolist()
    total      = len(label_list)

    groupe_to_idx = defaultdict(list)
    for i, lbl in enumerate(label_list):
        groupe_to_idx[lbl].append(i)

    groupes_multi  = [idx for idx in groupe_to_idx.values() if len(idx) > 1]
    paires_liees   = []
    paires_non_liees = []

    random.shuffle(groupes_multi)
    for indices in groupes_multi:
        if len(paires_liees) >= n // 2:
            break
        a, b = random.sample(indices, 2)
        paires_liees.append((a, b, "lie"))

    tentatives = 0
    while len(paires_non_liees) < n // 2 and tentatives < n * 20:
        a, b = random.sample(range(total), 2)
        if label_list[a] != label_list[b]:
            paires_non_liees.append((a, b, "non_lie"))
        tentatives += 1

    echantillon = paires_liees + paires_non_liees
    random.shuffle(echantillon)

    sortie = [
        {
            "nom_a":    noms_list[a],
            "nom_b":    noms_list[b],
            "groupe_a": label_list[a],
            "groupe_b": label_list[b],
            "langue_a": lang_list[a],
            "langue_b": lang_list[b],
            "geo_a":    geo_list[a],
            "geo_b":    geo_list[b],
            "attendu":  attendu,
            "valide":   None,
        }
        for a, b, attendu in echantillon
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)
    log.info("Évaluation : %d paires -> %s", len(sortie), path)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def exporter(data, labels, path_grouped, path_groupes):
    noms_grouped = []
    for i, item in enumerate(data):
        entry = {k: v for k, v in item.items() if not k.startswith("_")}
        entry["id_groupe_total"] = int(labels[i])
        noms_grouped.append(entry)

    with open(path_grouped, "w", encoding="utf-8") as f:
        json.dump(noms_grouped, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d noms)", path_grouped, len(noms_grouped))

    groupes = {}
    for entry in noms_grouped:
        gid       = entry["id_groupe_total"]
        origine   = entry.get("origine_brute", entry.get("origine", ""))
        id_g_prep = entry.get("id_groupe", entry["id"])
        if gid not in groupes:
            groupes[gid] = {"id_groupe_total": gid, "noms": [], "origines": [], "vus": set()}
        groupes[gid]["noms"].append(entry["nom"])
        if id_g_prep not in groupes[gid]["vus"] and origine:
            groupes[gid]["origines"].append(origine)
            groupes[gid]["vus"].add(id_g_prep)

    groupes_export = [
        {
            "id_groupe_total": g["id_groupe_total"],
            "noms":            sorted(g["noms"]),
            "origine_brute":   " --- ".join(g["origines"]),
        }
        for g in sorted(groupes.values(), key=lambda x: x["id_groupe_total"])
    ]

    with open(path_groupes, "w", encoding="utf-8") as f:
        json.dump(groupes_export, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d groupes)", path_groupes, len(groupes_export))


def log_stats(labels):
    from collections import Counter
    sizes      = Counter(labels.tolist())
    n_clusters = len(sizes)
    singletons = sum(1 for s in sizes.values() if s == 1)
    log.info(
        "Clusters=%d | singletons=%d (%.1f%%) | moy=%.1f | max=%d",
        n_clusters,
        singletons, 100 * singletons / n_clusters if n_clusters else 0,
        sum(sizes.values()) / n_clusters if n_clusters else 0,
        max(sizes.values()) if sizes else 0,
    )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    global SEUIL_LEV_PRESELECTION, SEUIL_SCORE_FINAL, LONGUEUR_MAX, PREFIXE_LEN, LEV_K, N_EVAL_PAIRES
    
    parser = argparse.ArgumentParser(description="Regroupement des noms avec JaroWinkler + TF-IDF.")
    parser.add_argument("--seuil_lev_preselection", type=float, default=SEUIL_LEV_PRESELECTION)
    parser.add_argument("--seuil_score_final", type=float, default=SEUIL_SCORE_FINAL)
    parser.add_argument("--longueur_max", type=int, default=LONGUEUR_MAX)
    parser.add_argument("--prefixe_len", type=int, default=PREFIXE_LEN)
    parser.add_argument("--lev_k", type=int, default=LEV_K)
    parser.add_argument("--n_eval_paires", type=int, default=N_EVAL_PAIRES)
    args = parser.parse_args()

    SEUIL_LEV_PRESELECTION = args.seuil_lev_preselection
    SEUIL_SCORE_FINAL = args.seuil_score_final
    LONGUEUR_MAX = args.longueur_max
    PREFIXE_LEN = args.prefixe_len
    LEV_K = args.lev_k
    N_EVAL_PAIRES = args.n_eval_paires

    log.info(f"Paramètres: LEV_PRE={SEUIL_LEV_PRESELECTION}, FINAL={SEUIL_SCORE_FINAL}, L_MAX={LONGUEUR_MAX}, PREF_L={PREFIXE_LEN}")

    if not os.path.isfile(INPUT_JSON):
        log.error("Fichier introuvable : %s", INPUT_JSON)
        return

    os.makedirs("data", exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info("%d entrées chargées", len(data))

    noms         = [d.get("nom", "")           for d in data]
    textes_bruts = [d.get("origine_brute", "") for d in data]
    origine     = [d.get("origine", "") for d in data]
    id_groupes   = [d.get("id_groupe", d.get("id", "")) for d in data]

    log.info("Extraction langue et géo...")
    langues = [extraire_langue(t) for t in tqdm(textes_bruts, desc="Langue", unit="nom")]
    geos    = [extraire_geo(t)    for t in tqdm(textes_bruts, desc="Géo",    unit="nom")]

    for i, d in enumerate(data):
        d["_langue"] = langues[i]
        d["_geo"]    = geos[i]

    log.info("Construction TF-IDF...")
    tfidf_dense, texte_to_idx = construire_tfidf(textes_bruts)

    paires_connues   = charger_cache(CACHE_PAIRES)
    nouvelles_paires = construire_paires(
        noms, id_groupes, textes_bruts, langues, geos,
        tfidf_dense, texte_to_idx, paires_connues,
    )
    
    # Validation NER des nouvelles paires
    nouvelles_paires_filtrees = filtrer_paires_ner(nouvelles_paires, noms)
    
    sauvegarder_cache(CACHE_PAIRES, paires_connues, nouvelles_paires_filtrees)

    labels = clusterer(noms, nouvelles_paires_filtrees, id_groupes)
    log_stats(labels)

    exporter(data, labels, OUTPUT_GROUPED, OUTPUT_GROUPES)
    generer_evaluation(data, labels, N_EVAL_PAIRES, OUTPUT_EVAL)


if __name__ == "__main__":
    main()