"""
regroupement_prenoms.py — Phase 2 prénoms
==========================================
Entrée  : prenoms/data/2_prenoms_clean.json
Sorties : prenoms/data/3_prenoms_grouped.json
          prenoms/data/3_groupes_prenoms.json
          prenoms/data/3_eval_prenoms.json
Caches  : prenoms/data/3_paires_prenoms.pkl
          prenoms/data/3_embeddings.npy

Architecture — 2 niveaux :

    NIVEAU 1 — Liens explicites (prenoms_lies, priorité absolue)
        Si B figure dans prenoms_lies de A (ou vice-versa), même groupe.
        Pas de seuil : liens éditoriaux déclarés dans la source.
        Filtre hub : citation > MAX_LIES_REF -> hub parasite, ignoré.

    NIVEAU 2 — Famille étymologique (CamemBERT + FAISS)
        Embedding sur histoire + etymologie + provenance (non lemmatisé).
        Score de validation :
            score = sim_sem + bonus_langue + bonus_geo
        Accepté si score >= SEUIL_SCORE_FINAL.
        Pas de terme pondéré par la richesse textuelle : ce terme donnait
        un bonus gratuit aux prénoms à texte court, causant les méga-groupes.
        Deux filtres anti-faux-positifs :
          - Textes identiques partagés par > SEUIL_TEXTE_GENERIQUE_N prénoms
          - Prénoms composés partageant le même radical (marie-xxx, jean-xxx)
          - Prénoms de longueur <= MIN_LEN_FUSION exclus des fusions FAISS
            (embeddings non représentatifs, source de ponts transitifs)

    UNION-FIND :
        Chaque prénom appartient à exactement un groupe.
        N1 injecté en premier, N2 en second.

Dépendances :
    pip install sentence-transformers numpy tqdm faiss-cpu
    (remplacer faiss-cpu par faiss-gpu si GPU disponible — incompatibles)
"""

import json
import logging
import os
import pickle
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_DISPONIBLE = True
except ImportError:
    FAISS_DISPONIBLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_JSON       = "prenoms/data/2_prenoms_clean.json"
OUTPUT_GROUPED   = "prenoms/data/3_prenoms_grouped.json"
OUTPUT_GROUPES   = "prenoms/data/3_groupes_prenoms.json"
OUTPUT_EVAL      = "prenoms/data/3_eval_prenoms.json"
CACHE_PAIRES     = "prenoms/data/3_paires_prenoms.pkl"
CACHE_EMBEDDINGS = "prenoms/data/3_embeddings.npy"

MODEL_NAME = "Lajavaness/sentence-camembert-base"

# Niveau 1 — filtre hub
MAX_LIES_REF = 8

# Niveau 2 — FAISS présélection
# Seuil de présélection FAISS : on récupère tout ce qui est >= SEUIL_FAISS,
# le scoring final applique ensuite SEUIL_SCORE_FINAL sur le score composite.
# SEUIL_FAISS légèrement inférieur à SEUIL_SCORE_FINAL pour laisser les
# bonuses langue/geo faire leur travail.
SEUIL_FAISS    = 0.88
K_VOISINS_SEM  = 50

# Score composite : sim_sem + bonus_langue + bonus_geo >= SEUIL_SCORE_FINAL
# bonus_langue = BONUS_LANGUE si même langue non vide
# bonus_geo    = BONUS_GEO    si même geo non vide
# Valeurs calibrées pour que deux prénoms de même famille étymologique
# (sim ~0.90) + même langue passent, mais pas des prénoms de familles
# différentes avec sim ~0.88 sans contexte commun.
SEUIL_SCORE_FINAL = 0.92
BONUS_LANGUE      = 0.04
BONUS_GEO         = 0.02

# Filtres anti-faux-positifs
SEUIL_TEXTE_GENERIQUE_N = 20  # texte partagé par > N prénoms
SEUIL_RADICAL_N         = 5   # radical composé partagé par >= N prénoms

# Longueur minimale (chars) du prénom pour participer aux fusions FAISS.
# "il", "me", "ils", "abd" → exclus.
MIN_LEN_FUSION = 4

# Nombre minimum de tokens du texte CamemBERT pour autoriser une fusion FAISS.
# Un texte d'un seul mot ("Tahiti", "Réunion", "Lion", "pays") produit un
# embedding non discriminant : tous ces prénoms se retrouvent proches en
# espace cosine uniquement parce que leur texte est un mot isolé, pas parce
# qu'ils partagent une famille étymologique.
# Les deux membres de la paire doivent dépasser ce seuil.
MIN_TOKENS_TEXTE = 5

# Mots grammaticaux et artefacts de scraping à rejeter des prenoms_lies.
# "etant", "dont", "qui", "les", "des" apparaissent dans les listes de liens
# scrappées sur prenoms.com quand la page HTML est mal parsée.
# Un prénom valide : lettres uniquement (accents OK), longueur >= 2,
# pas dans cette liste de stopwords.
STOPWORDS_LIES = {
    "etant", "dont", "qui", "que", "les", "des", "une", "son", "ses",
    "par", "sur", "est", "pas", "plus", "tout", "cette", "dans", "pour",
    "avec", "comme", "mais", "donc", "car", "ainsi", "aussi", "tres",
    "saint", "sainte", "saints", "saintes",
}

N_EVAL_PAIRES = 200


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    """Un prénom = un groupe. Path compression + union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def meme_groupe(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def labels(self) -> np.ndarray:
        roots   = {self.find(i) for i in range(len(self.parent))}
        root_id = {r: idx + 1 for idx, r in enumerate(sorted(roots))}
        return np.array([root_id[self.find(i)] for i in range(len(self.parent))])


# ---------------------------------------------------------------------------
# NIVEAU 1 — Liens explicites (prenoms_lies)
# ---------------------------------------------------------------------------

import unicodedata
import re as _re

def _est_prenom_valide(p: str) -> bool:
    """
    Valide qu'une entrée de prenoms_lies est bien un prénom et non un artefact
    de scraping. Critères : lettres uniquement (accents OK), longueur >= 2,
    pas dans STOPWORDS_LIES.
    Exemple rejeté : "etant", "dont", "saint", "les".
    """
    # Normalise les accents pour la comparaison avec STOPWORDS_LIES
    p_norm = unicodedata.normalize("NFD", p.lower())
    p_norm = _re.sub(r"[\u0300-\u036f]", "", p_norm)  # supprime diacritiques
    if p_norm in STOPWORDS_LIES:
        return False
    # Doit contenir uniquement des lettres et tirets/espaces (prénom composé OK)
    if not _re.match(r"^[a-zàâäéèêëîïôùûüÿœæç\- ]+$", p.lower()):
        return False
    if len(p.strip()) < 2:
        return False
    return True


def construire_paires_explicites(data: list, prenom_to_idx: dict) -> list:
    """
    Pour chaque prénom A avec prenoms_lies=[B,C,...], crée (A,B), (A,C) etc.

    Filtre hub : prénom cité par > MAX_LIES_REF sources = hub parasite.

    Filtre artefact : les entrées de prenoms_lies qui sont des mots
    grammaticaux ou des artefacts de scraping ("etant", "saint", "dont"...)
    sont rejetées via _est_prenom_valide(). Le chemin de fusion
    mariame -> etant -> tahiti -> ... était causé par ce type d'artefact.

    Retourne liste de (i, j) sans score — liens absolus.
    """
    lies_freq: dict = defaultdict(int)
    for d in data:
        for p in d.get("prenoms_lies", []):
            lies_freq[p] += 1

    n_hubs = sum(1 for v in lies_freq.values() if v > MAX_LIES_REF)
    log.info("Hubs filtres (freq > %d) : %d", MAX_LIES_REF, n_hubs)

    paires      = []
    vus         = set()
    n_hors      = 0
    n_artefacts = 0

    for i, d in enumerate(data):
        for p_lie in d.get("prenoms_lies", []):
            if lies_freq[p_lie] > MAX_LIES_REF:
                continue
            if not _est_prenom_valide(p_lie):
                n_artefacts += 1
                continue
            j = prenom_to_idx.get(p_lie)
            if j is None:
                n_hors += 1
                continue
            if i == j:
                continue
            cle = (min(i, j), max(i, j))
            if cle not in vus:
                vus.add(cle)
                paires.append((i, j))

    log.info(
        "Paires N1 : %d | hors corpus : %d | artefacts rejetes : %d",
        len(paires), n_hors, n_artefacts,
    )
    return paires


# ---------------------------------------------------------------------------
# NIVEAU 2 — Sémantique (CamemBERT + FAISS)
# ---------------------------------------------------------------------------

def construire_texte_camembert(item: dict) -> str:
    """
    histoire + etymologie + provenance concaténés, non lemmatisés.
    signification et texte_brut exclus.
    """
    parties = []
    for champ in ("histoire", "etymologie", "provenance"):
        val = item.get(champ)
        if val and isinstance(val, str) and val.strip():
            parties.append(val.strip())
    return " ".join(parties)


def charger_ou_calculer_embeddings(textes: list) -> np.ndarray:
    if os.path.isfile(CACHE_EMBEDDINGS):
        embs = np.load(CACHE_EMBEDDINGS)
        if embs.shape[0] == len(textes):
            log.info("Embeddings charges depuis cache (%s)", CACHE_EMBEDDINGS)
            return embs
        log.warning("Cache invalide (%d vs %d) — recalcul", embs.shape[0], len(textes))
    return _calculer_embeddings(textes)


def _calculer_embeddings(textes: list) -> np.ndarray:
    """
    Déduplication avant encode : économie ~30% sur les textes partagés.
    L2-normalisé : cosine = dot product (requis par FAISS IndexFlatIP).
    """
    textes_clean   = [t if t and t.strip() else "__vide__" for t in textes]
    textes_uniques = list(dict.fromkeys(textes_clean))
    texte_to_idx   = {t: i for i, t in enumerate(textes_uniques)}

    log.info(
        "Embeddings : %d uniques / %d total (%d doublons)",
        len(textes_uniques), len(textes), len(textes) - len(textes_uniques),
    )
    model     = SentenceTransformer(MODEL_NAME)
    embs_uniq = model.encode(
        textes_uniques,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embs = np.stack([embs_uniq[texte_to_idx[t]] for t in textes_clean])
    log.info("Matrice embeddings : %s", embs.shape)
    embs = embs.astype(np.float32)
    np.save(CACHE_EMBEDDINGS, embs)
    log.info("Embeddings sauvegardes : %s", CACHE_EMBEDDINGS)
    return embs


def detecter_textes_generiques(textes: list, prenoms: list) -> tuple:
    """
    TYPE 1 — Textes identiques partagés par > SEUIL_TEXTE_GENERIQUE_N prénoms.
        Paires rejetées si les deux membres portent un tel texte.

    TYPE 2 — Prénoms composés partageant le même radical (marie-xxx, jean-xxx).
        Sim cosine structurellement élevée sans lien étymologique réel.
        Paires rejetées si les deux membres partagent le radical bloqué.

    Retourne (indices_gen: set, prefixes_bloc: set).
    """
    # Type 1
    freq        = Counter(t.strip() for t in textes if t and t.strip())
    generiques  = {t for t, c in freq.items() if c > SEUIL_TEXTE_GENERIQUE_N}
    indices_gen = {i for i, t in enumerate(textes) if t and t.strip() in generiques}
    log.info("Textes generiques (type 1) : %d textes -> %d prenoms",
             len(generiques), len(indices_gen))

    # Type 2
    radical_freq: dict = defaultdict(int)
    for p in prenoms:
        tokens = p.split()
        if len(tokens) >= 2 and len(tokens[0]) >= 4:
            radical_freq[tokens[0]] += 1

    prefixes_bloc     = {r for r, c in radical_freq.items() if c >= SEUIL_RADICAL_N}
    n_prenoms_radical = sum(
        1 for p in prenoms
        if len(p.split()) >= 2 and p.split()[0] in prefixes_bloc
    )
    log.info("Radicaux bloques (type 2) : %s -> %d prenoms",
             sorted(prefixes_bloc), n_prenoms_radical)

    return indices_gen, prefixes_bloc


def radical_bloque(p_i: str, p_j: str, prefixes_bloc: set) -> bool:
    ti, tj = p_i.split(), p_j.split()
    if len(ti) < 2 or len(tj) < 2:
        return False
    return ti[0] == tj[0] and ti[0] in prefixes_bloc


def _nb_tokens(texte: str) -> int:
    """Nombre de tokens whitespace dans le texte CamemBERT source."""
    return len(texte.split()) if texte and texte.strip() else 0


def _voisins_faiss(embs: np.ndarray, uf: UnionFind, prenoms: list,
                   textes: list, indices_gen: set, prefixes_bloc: set,
                   k: int) -> list:
    """
    Présélection FAISS >= SEUIL_FAISS avec filtres anti-faux-positifs.

    Filtre MIN_LEN_FUSION : longueur de prénom.
    Filtre MIN_TOKENS_TEXTE : les deux membres doivent avoir un texte
        d'au moins MIN_TOKENS_TEXTE tokens. Un texte d'un seul mot
        ("Tahiti", "Réunion", "Lion") produit un embedding non discriminant
        — tous ces prénoms se retrouvent proches uniquement parce que leur
        texte est un mot isolé, causant des ponts transitifs parasites.
    """
    nb_tok = [_nb_tokens(t) for t in textes]

    n     = embs.shape[0]
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    scores, indices = index.search(embs, k + 1)

    paires      = []
    vus         = set()
    n_groupe    = 0
    n_gen       = 0
    n_rad       = 0
    n_court     = 0
    n_pauvre    = 0

    for i in range(n):
        for rank in range(1, k + 1):
            j   = int(indices[i, rank])
            sim = float(scores[i, rank])
            if sim < SEUIL_FAISS:
                break
            if uf.meme_groupe(i, j):
                n_groupe += 1
                continue
            if len(prenoms[i]) <= MIN_LEN_FUSION or len(prenoms[j]) <= MIN_LEN_FUSION:
                n_court += 1
                continue
            if nb_tok[i] < MIN_TOKENS_TEXTE or nb_tok[j] < MIN_TOKENS_TEXTE:
                n_pauvre += 1
                continue
            if i in indices_gen and j in indices_gen:
                n_gen += 1
                continue
            if radical_bloque(prenoms[i], prenoms[j], prefixes_bloc):
                n_rad += 1
                continue
            cle = (min(i, j), max(i, j))
            if cle not in vus:
                vus.add(cle)
                paires.append((i, j, sim))

    log.info(
        "Candidats FAISS (>= %.2f) : %d | deja groupes : %d | "
        "courts (<=%d) : %d | texte pauvre (<%d tok) : %d | gen : %d | radical : %d",
        SEUIL_FAISS, len(paires), n_groupe,
        MIN_LEN_FUSION, n_court, MIN_TOKENS_TEXTE, n_pauvre, n_gen, n_rad,
    )
    return paires


def _voisins_brute(embs: np.ndarray, uf: UnionFind, prenoms: list,
                   textes: list, indices_gen: set, prefixes_bloc: set,
                   k: int, bloc: int = 256) -> list:
    """Fallback sans FAISS. Mêmes filtres que _voisins_faiss."""
    nb_tok = [_nb_tokens(t) for t in textes]
    n      = embs.shape[0]
    paires = []
    vus    = set()

    for i_start in tqdm(range(0, n, bloc), desc="Score sem (brute)", unit="bloc"):
        i_end     = min(i_start + bloc, n)
        sims_bloc = embs[i_start:i_end] @ embs.T

        for local_i in range(i_end - i_start):
            gi   = i_start + local_i
            row  = sims_bloc[local_i]
            k_e  = min(k, n - 1)
            topk = np.argpartition(row, -k_e)[-k_e:]
            for j in topk:
                j   = int(j)
                sim = float(row[j])
                if sim < SEUIL_FAISS or j <= gi:
                    continue
                if uf.meme_groupe(gi, j):
                    continue
                if len(prenoms[gi]) <= MIN_LEN_FUSION or len(prenoms[j]) <= MIN_LEN_FUSION:
                    continue
                if nb_tok[gi] < MIN_TOKENS_TEXTE or nb_tok[j] < MIN_TOKENS_TEXTE:
                    continue
                if gi in indices_gen and j in indices_gen:
                    continue
                if radical_bloque(prenoms[gi], prenoms[j], prefixes_bloc):
                    continue
                cle = (gi, j)
                if cle not in vus:
                    vus.add(cle)
                    paires.append((gi, j, sim))

    log.info("Candidats brute-force : %d", len(paires))
    return paires


def scorer_paires_sem(paires: list, langues: list, geos: list) -> list:
    """
    Scoring composite sur les candidats FAISS :
        score = sim_sem + bonus_langue + bonus_geo

    Besoin : remplacer la formule (1-t)*1.0 + t*score_ctx qui donnait
    un bonus gratuit aux prénoms à texte court (t faible -> score ~ 1.0
    quelle que soit la sim). Cette formule était la cause des méga-groupes.

    La sim_sem seule porte le signal étymologique. Les bonuses langue/geo
    sont des signaux contextuels complémentaires, pas des substituts.

    Seuil : SEUIL_SCORE_FINAL = 0.92 par défaut.
    Plage : sim_sem ∈ [SEUIL_FAISS=0.88, 1.0]
            bonus_langue ∈ {0, BONUS_LANGUE=0.04}
            bonus_geo    ∈ {0, BONUS_GEO=0.02}
    Score max possible : 1.0 + 0.04 + 0.02 = 1.06
    Score min pour passer : 0.92
    -> une paire avec sim=0.90 + même langue passe (0.94 >= 0.92)
    -> une paire avec sim=0.88 sans contexte commun ne passe pas
    """
    retenues = []
    for i, j, sim_sem in paires:
        b_langue = BONUS_LANGUE if langues[i] and langues[i] == langues[j] else 0.0
        b_geo    = BONUS_GEO    if geos[i]    and geos[i]    == geos[j]    else 0.0
        score    = sim_sem + b_langue + b_geo
        if score >= SEUIL_SCORE_FINAL:
            retenues.append((i, j, score))

    log.info(
        "Paires N2 apres scoring : %d / %d candidats (seuil=%.2f)",
        len(retenues), len(paires), SEUIL_SCORE_FINAL,
    )
    return retenues


# ---------------------------------------------------------------------------
# Cache paires
# ---------------------------------------------------------------------------

def sauvegarder_cache_paires(path: str, paires_n2: list):
    toutes = {(min(i, j), max(i, j)) for i, j, _ in paires_n2}
    with open(path, "wb") as f:
        pickle.dump(toutes, f)
    log.info("Cache paires sauvegarde : %d paires", len(toutes))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def exporter(data: list, labels: np.ndarray,
             path_grouped: str, path_groupes: str):
    prenoms_grouped = []
    for i, item in enumerate(data):
        entry = dict(item)
        entry["id_groupe_total"] = int(labels[i])
        prenoms_grouped.append(entry)

    with open(path_grouped, "w", encoding="utf-8") as f:
        json.dump(prenoms_grouped, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d prenoms)", path_grouped, len(prenoms_grouped))

    groupes: dict = {}
    for entry in prenoms_grouped:
        gid = entry["id_groupe_total"]
        if gid not in groupes:
            groupes[gid] = {"id_groupe_total": gid, "prenoms": [],
                            "textes": [], "vus": set()}
        groupes[gid]["prenoms"].append(entry["prenom"])
        tb = entry.get("texte_brut", "")
        if tb and tb not in groupes[gid]["vus"]:
            groupes[gid]["textes"].append(tb)
            groupes[gid]["vus"].add(tb)

    groupes_export = [
        {
            "id_groupe_total": g["id_groupe_total"],
            "prenoms":         sorted(g["prenoms"]),
            "texte_brut":      " --- ".join(g["textes"]),
        }
        for g in sorted(groupes.values(), key=lambda x: x["id_groupe_total"])
    ]

    with open(path_groupes, "w", encoding="utf-8") as f:
        json.dump(groupes_export, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d groupes)", path_groupes, len(groupes_export))


def log_stats(labels: np.ndarray):
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
    log.info("Top 5 groupes : %s", sizes.most_common(5))


def generer_evaluation(data: list, labels: np.ndarray, n: int, path: str):
    prenoms_list = [d["prenom"] for d in data]
    label_list   = labels.tolist()
    total        = len(label_list)

    grp_to_idx = defaultdict(list)
    for i, lbl in enumerate(label_list):
        grp_to_idx[lbl].append(i)

    groupes_multi    = [idx for idx in grp_to_idx.values() if len(idx) > 1]
    paires_liees     = []
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
            "prenom_a": prenoms_list[a],
            "prenom_b": prenoms_list[b],
            "groupe_a": label_list[a],
            "groupe_b": label_list[b],
            "langue_a": data[a].get("langue", ""),
            "langue_b": data[b].get("langue", ""),
            "attendu":  attendu,
            "valide":   None,
        }
        for a, b, attendu in echantillon
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)
    log.info("Evaluation : %d paires -> %s", len(sortie), path)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    global SEUIL_FAISS, K_VOISINS_SEM, SEUIL_SCORE_FINAL
    global BONUS_LANGUE, BONUS_GEO, N_EVAL_PAIRES
    global MAX_LIES_REF, SEUIL_TEXTE_GENERIQUE_N, SEUIL_RADICAL_N
    global MIN_LEN_FUSION, MIN_TOKENS_TEXTE

    parser = argparse.ArgumentParser(
        description="Regroupement prenoms : N1 lies > N2 CamemBERT+FAISS."
    )
    parser.add_argument("--seuil_faiss",             type=float, default=SEUIL_FAISS)
    parser.add_argument("--seuil_score_final",       type=float, default=SEUIL_SCORE_FINAL)
    parser.add_argument("--bonus_langue",            type=float, default=BONUS_LANGUE)
    parser.add_argument("--bonus_geo",               type=float, default=BONUS_GEO)
    parser.add_argument("--k_voisins_sem",           type=int,   default=K_VOISINS_SEM)
    parser.add_argument("--max_lies_ref",            type=int,   default=MAX_LIES_REF)
    parser.add_argument("--seuil_texte_generique_n", type=int,   default=SEUIL_TEXTE_GENERIQUE_N)
    parser.add_argument("--seuil_radical_n",         type=int,   default=SEUIL_RADICAL_N)
    parser.add_argument("--min_len_fusion",          type=int,   default=MIN_LEN_FUSION)
    parser.add_argument("--min_tokens_texte",        type=int,   default=MIN_TOKENS_TEXTE)
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Supprime les caches embeddings et paires avant de recalculer.",
    )
    args = parser.parse_args()

    SEUIL_FAISS             = args.seuil_faiss
    SEUIL_SCORE_FINAL       = args.seuil_score_final
    BONUS_LANGUE            = args.bonus_langue
    BONUS_GEO               = args.bonus_geo
    K_VOISINS_SEM           = args.k_voisins_sem
    MAX_LIES_REF            = args.max_lies_ref
    SEUIL_TEXTE_GENERIQUE_N = args.seuil_texte_generique_n
    SEUIL_RADICAL_N         = args.seuil_radical_n
    MIN_LEN_FUSION          = args.min_len_fusion
    MIN_TOKENS_TEXTE        = args.min_tokens_texte

    log.info(
        "Parametres : FAISS=%.2f FINAL=%.2f BL=%.2f BG=%.2f K=%d "
        "MAX_LIES=%d GEN_N=%d RAD_N=%d MIN_LEN=%d MIN_TOK=%d | FAISS_dispo=%s",
        SEUIL_FAISS, SEUIL_SCORE_FINAL, BONUS_LANGUE, BONUS_GEO,
        K_VOISINS_SEM, MAX_LIES_REF, SEUIL_TEXTE_GENERIQUE_N, SEUIL_RADICAL_N,
        MIN_LEN_FUSION, MIN_TOKENS_TEXTE, "oui" if FAISS_DISPONIBLE else "non",
    )

    if not os.path.isfile(INPUT_JSON):
        log.error("Fichier introuvable : %s", INPUT_JSON)
        return

    os.makedirs(os.path.dirname(OUTPUT_GROUPED), exist_ok=True)

    if args.force_recompute:
        for chemin in (CACHE_EMBEDDINGS, CACHE_PAIRES):
            if os.path.isfile(chemin):
                os.remove(chemin)
                log.info("Cache supprime : %s", chemin)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info("%d prenoms charges", len(data))

    prenoms       = [d["prenom"] for d in data]
    prenom_to_idx = {p: i for i, p in enumerate(prenoms)}
    langues       = [d.get("langue", "") for d in data]
    geos          = [d.get("geo",    "") for d in data]

    uf = UnionFind(len(prenoms))

    # Groupes Phase 1 (id_groupe attribué par prepare_prenoms)
    id_groupes_p1 = [
        d.get("id_groupe") or d.get("id") or str(i)
        for i, d in enumerate(data)
    ]
    grp_p1: dict = defaultdict(list)
    for i, gid in enumerate(id_groupes_p1):
        grp_p1[gid].append(i)
    n_inj_p1 = 0
    for indices in grp_p1.values():
        if len(indices) > 1:
            for k in range(1, len(indices)):
                uf.union(indices[0], indices[k])
            n_inj_p1 += 1
    log.info("Groupes Phase 1 injectes : %d groupes multi-membres", n_inj_p1)

    # -----------------------------------------------------------------------
    # NIVEAU 1
    # -----------------------------------------------------------------------
    log.info("=== NIVEAU 1 : liens explicites (prenoms_lies) ===")
    paires_n1 = construire_paires_explicites(data, prenom_to_idx)
    n_fus_n1  = sum(1 for i, j in paires_n1 if uf.union(i, j))
    log.info("Fusions N1 : %d / %d paires", n_fus_n1, len(paires_n1))

    # -----------------------------------------------------------------------
    # NIVEAU 2
    # -----------------------------------------------------------------------
    log.info("=== NIVEAU 2 : famille etymologique (CamemBERT + FAISS) ===")
    textes_cam             = [construire_texte_camembert(d) for d in data]
    embs                   = charger_ou_calculer_embeddings(textes_cam)
    indices_gen, pref_bloc = detecter_textes_generiques(textes_cam, prenoms)

    if FAISS_DISPONIBLE:
        candidats = _voisins_faiss(embs, uf, prenoms, textes_cam, indices_gen, pref_bloc, K_VOISINS_SEM)
    else:
        log.warning("FAISS non disponible — fallback brute-force")
        candidats = _voisins_brute(embs, uf, prenoms, textes_cam, indices_gen, pref_bloc, K_VOISINS_SEM)

    paires_n2 = scorer_paires_sem(candidats, langues, geos)
    n_fus_n2  = sum(1 for i, j, _ in paires_n2 if uf.union(i, j))
    log.info("Fusions N2 : %d / %d paires", n_fus_n2, len(paires_n2))

    sauvegarder_cache_paires(CACHE_PAIRES, paires_n2)

    log.info("Bilan : N1=%d | N2=%d | total=%d", n_fus_n1, n_fus_n2, n_fus_n1 + n_fus_n2)

    labels = uf.labels()
    log_stats(labels)

    exporter(data, labels, OUTPUT_GROUPED, OUTPUT_GROUPES)
    generer_evaluation(data, labels, N_EVAL_PAIRES, OUTPUT_EVAL)


if __name__ == "__main__":
    main()