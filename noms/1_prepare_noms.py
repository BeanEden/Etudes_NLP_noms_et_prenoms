"""
Préparation noms_clean.json
============================
Entrées : ./data/names.json    -> [{name, origins: [OID, ...]}, ...]
          ./data/origins.json  -> {OID: "texte explicatif", ...}
Sortie  : ./data/noms_clean.json

Logique noms_lies :
    Pour chaque nom N et son texte d'origine T(N) :
    1. Regex : sections "Variantes :", "Dérivés :" dans T(N)
    2. Levenshtein : mots capitalisés de T(N) dont la distance avec N <= seuil adaptatif

Installation :
    pip install spacy rapidfuzz
    python -m spacy download fr_core_news_sm
"""

import hashlib
import json
import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import spacy
from rapidfuzz.distance import Levenshtein as Lev

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_NAMES   = "noms/data/0_names.json"
INPUT_ORIGINS = "noms/data/0_origins.json"
OUTPUT_JSON   = "noms/data/1_noms_clean.json"

SPACY_MODEL   = "fr_core_news_sm"
MIN_TOKEN_LEN = 3
SEPARATEUR_ORIGINES = " "
MAX_NOMS_LIES = 15
NB_WORKERS    = None   # None = os.cpu_count()

STOPWORDS_METIER = {
    "prénom", "prenom", "personne", "personnes", "notamment", "ainsi",
    "également", "aussi", "très", "plus", "bien", "peu", "tout", "tous",
    "cette", "celui", "celle", "ceux", "celles", "faire", "être", "avoir",
    "porter", "donner", "venir", "vouloir", "dire", "voir", "aller",
    "saint", "sainte", "fête", "fete", "jour", "siècle", "siecle",
    "ans", "année", "annee", "environ", "vers", "depuis", "jusqu",
    "nom", "noms", "famille",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def normaliser(nom: str) -> str:
    """
    Normalisation canonique partagée par toutes les fonctions de recherche.
    Tirets remplacés par espace (pas supprimés) pour que nom_original restitue
    "Saint Martin" depuis "saint martin" via capitalize par mot.
    """
    nom = unicodedata.normalize("NFD", nom)
    nom = "".join(c for c in nom if unicodedata.category(c) != "Mn")
    nom = nom.lower().replace("-", " ")
    return " ".join(nom.split())   # normalise les espaces multiples


def generer_id(nom: str) -> str:
    """ID déterministe sur le nom normalisé — stable entre runs."""
    return hashlib.md5(normaliser(nom).encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Extraction regex : sections Variantes / Dérivés
# ---------------------------------------------------------------------------

PATTERNS_VARIANTES = [
    r"variantes?\s+ou\s+formes?\s+voisines?\s*:",
    r"formes?\s+voisines?\s+ou\s+variantes?\s*:",
    r"variantes?\s*:",
    r"formes?\s+voisines?\s*:",
    r"formes?\s+apparentées?\s*:",
]
PATTERNS_DERIVES = [
    r"dérivés?\s+ou\s+formes?\s+dérivées?\s*:",
    r"formes?\s+dérivées?\s+ou\s+dérivés?\s*:",
    r"dérivés?\s*:",
    r"formes?\s+dérivées?\s*:",
]
SEP_NOMS    = re.compile(r"[,;]+")
FIN_SECTION = re.compile(
    r"(?:variantes?|dérivés?|formes?|diminutifs?|hypocoristiques?"
    r"|fréquence|étymologie|origine|histoire|note\s*:)",
    re.IGNORECASE,
)


def _extraire_liste_apres_label(texte: str, patterns: list) -> list:
    texte_lower = texte.lower()
    for pattern in patterns:
        match = re.search(pattern, texte_lower, re.IGNORECASE)
        if not match:
            continue
        segment = texte[match.end():]
        fin = re.search(r"\.\s+[A-Z]|\.\s*$", segment)
        if fin:
            segment = segment[:fin.start()]
        fin_lbl = FIN_SECTION.search(segment)
        if fin_lbl:
            segment = segment[:fin_lbl.start()]
        segment = re.sub(r"\([^)]*\)", "", segment)
        noms = [t.strip().strip(".") for t in SEP_NOMS.split(segment)]
        noms = [n for n in noms if len(n) >= 2 and not n.isdigit()]
        if noms:
            return noms
    return []


def extraire_noms_regex(texte: str) -> list:
    """Variantes + dérivés étiquetés, fusionnés sans doublon."""
    vus, fusion = set(), []
    for n in (_extraire_liste_apres_label(texte, PATTERNS_VARIANTES) +
              _extraire_liste_apres_label(texte, PATTERNS_DERIVES)):
        cle = n.lower().strip()
        if cle not in vus:
            fusion.append(n)
            vus.add(cle)
    return fusion


# ---------------------------------------------------------------------------
# Extraction Levenshtein : candidats depuis le texte brut
# ---------------------------------------------------------------------------

# Mots capitalisés simples : "Schwartz", "Pojan"
RE_MOT_MAJUSCULE = re.compile(r"\b([A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ][A-Za-zÀ-ÿ\-]{1,})\b")

# Tokens entre marqueurs typographiques : backticks, guillemets anglais, italique wiki
# Capture les termes techniques écrits en minuscule : `âbid, `abd, *abid*, _abd_
RE_BALISE = re.compile(r"[`*_\"'']([A-Za-zÀ-ÿ\-âêîôûäëïöü]{2,})[`*_\"'']")

# Particules de noms composés : uniquement en majuscule (Da, De, Di, Du, Le, La, Van, Von...)
# Sans re.IGNORECASE : "de Pojan" ne matche pas, "Da Pojan" oui.
RE_PARTICULE_NOM = re.compile(
    r"\b((?:D[aeiou]|D[e]s?|D[u]|L[ae]s?|V[ao]n|D[i]|D[e]l(?:la)?|A[f]|O'|M[ao]c|S[ao]n(?:ta)?)"
    r"\s+[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ][A-Za-zÀ-ÿ\-]{1,}(?:\s+[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ][A-Za-zÀ-ÿ\-]{1,})?)\b"
)


def extraire_candidats(texte_brut: str) -> list:
    """
    Collecte les candidats depuis trois sources complémentaires :

    1. Mots capitalisés (source principale) :
       "...également écrit Da Poian..." -> ["Da", "Poian"]
       Couvre les noms propres écrits normalement.

    2. Tokens balisés (backticks, guillemets, italique) :
       "...formé sur `âbid ou sur `abd..." -> ["âbid", "abd"]
       Couvre les termes techniques en minuscule signalés typographiquement.

    3. Noms à particule (séquences particule + majuscule) :
       "...également écrit Da Poian..." -> ["Da Poian"]
       Couvre les noms composés avec particule.

    Retourne une liste de chaînes candidates (casse originale préservée).
    """
    candidats = []
    vus_norm  = set()

    def ajouter(c: str) -> None:
        n = normaliser(c)
        if n and n not in vus_norm:
            candidats.append(c)
            vus_norm.add(n)

    for m in RE_MOT_MAJUSCULE.findall(texte_brut):
        ajouter(m)
    for m in RE_BALISE.findall(texte_brut):
        ajouter(m)
    for m in RE_PARTICULE_NOM.findall(texte_brut):
        ajouter(m)

    return candidats


def seuil_distance(longueur: int) -> int:
    """
    Distance Levenshtein absolue maximale selon la longueur du nom normalisé.
        <= 4 chars : seuil 1  (Aaron/Aron : dist=1)
        <= 7 chars : seuil 2  (Aabadi/Abdi : dist=2, Schwarz/Schwartz : dist=1)
        >  7 chars : seuil 3  (Schwartstein/Schwarzstein : dist=2)
    """
    if longueur <= 4:
        return 1
    if longueur <= 7:
        return 2
    return 3


def extraire_noms_lies_lev(nom: str, texte_brut: str) -> list:
    """
    Besoin : trouver dans le texte brut les candidats proches du nom source.

    Logique :
        1. Collecter les candidats via extraire_candidats (3 sources).
        2. Pour chaque candidat, calculer la distance Levenshtein entre
           le nom normalisé et le candidat normalisé.
        3. Garder les candidats dont la distance <= seuil adaptatif.
    """
    nom_norm = normaliser(nom)
    seuil    = seuil_distance(len(nom_norm))
    vus      = {nom_norm}
    noms_lies = []

    for candidat in extraire_candidats(texte_brut):
        cand_norm = normaliser(candidat)
        if cand_norm in vus:
            continue
        dist = Lev.distance(nom_norm, cand_norm, score_cutoff=seuil)
        if dist is not None and dist <= seuil:
            noms_lies.append(candidat)
            vus.add(cand_norm)

    return noms_lies


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def nettoyer_nom_lie(nom: str) -> str:
    """Retire chiffres et ponctuation, normalise."""
    nom = re.sub(r"\([^)]*\)", "", nom) 
    nom = re.sub(r"[\d]", "", nom)
    nom = re.sub(r"[^\w\s\-]", "", nom)
    return normaliser(nom.strip())


def nom_original(nom_norm: str) -> str:
    """Version présentable : capitalize par mot. "saint martin" -> "Saint Martin"."""
    return " ".join(mot.capitalize() for mot in nom_norm.split())


def fusionner_noms_lies(nom_source: str, *sources) -> list:
    """Union ordonnée de toutes les sources, sans le nom source, sans doublon."""
    nom_source_norm = normaliser(nom_source)
    vus, liste = {nom_source_norm}, []
    for source in sources:
        for n in source:
            cle = normaliser(n)
            if cle not in vus:
                liste.append(n)
                vus.add(cle)
    return liste[:MAX_NOMS_LIES]


# ---------------------------------------------------------------------------
# Nettoyage NLP
# ---------------------------------------------------------------------------

def nettoyer_texte(texte: str) -> str:
    if not isinstance(texte, str) or not texte.strip():
        return ""
    texte = unicodedata.normalize("NFKC", texte)
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = re.sub(r"https?://\S+|www\.\S+", " ", texte)
    texte = re.sub(r"\b\d+\w*\b", " ", texte)
    texte = re.sub(r"[^\w\s\'\-]", " ", texte)
    texte = re.sub(r"(?<!\w)-|-(?!\w)", " ", texte)
    texte = texte.lower()
    return re.sub(r"\s+", " ", texte).strip()


def lemmatiser(texte: str, nlp: spacy.Language) -> str:
    if not texte:
        return ""
    tokens = []
    for token in nlp(texte):
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemme = token.lemma_.lower().strip()
        if len(lemme) < MIN_TOKEN_LEN or lemme in STOPWORDS_METIER:
            continue
        tokens.append(lemme)
    return " ".join(tokens)


def traiter_texte_origine(texte: str, nlp: spacy.Language) -> str:
    return lemmatiser(nettoyer_texte(texte), nlp)


# ---------------------------------------------------------------------------
# Jointure
# ---------------------------------------------------------------------------

def joindre_origines(origins_ids: list, origins_map: dict) -> str:
    textes = []
    for oid in origins_ids:
        texte = origins_map.get(oid, "")
        if texte.strip():
            textes.append(texte.strip())
        else:
            log.warning("ID origine introuvable ou vide : %s", oid)
    return SEPARATEUR_ORIGINES.join(textes)


# ---------------------------------------------------------------------------
# Traitement unitaire (un nom, appelé par chaque thread)
# ---------------------------------------------------------------------------

def traiter_un_nom(item: dict, origins_map: dict, nlp: spacy.Language) -> dict:
    """
    Unité de travail pour ThreadPoolExecutor.
    origins_map, nlp et les fonctions appelées sont en lecture seule -> thread-safe.

    noms_lies est retourné normalisé (même format que names.json) dès ici.
    La résolution des id_noms_lies se fait en post-traitement dans main(),
    une fois que tous les IDs sont connus (y compris les nouveaux noms découverts).
    """
    nom        = item.get("name", "").strip()
    origin_ids = item.get("origins", [])
    texte_brut = joindre_origines(origin_ids, origins_map)

    noms_regex = extraire_noms_regex(texte_brut)
    noms_lev   = extraire_noms_lies_lev(nom, texte_brut)
    # Normalisation + nettoyage ponctuation/chiffres sur chaque nom lié
    noms_lies = [
        nettoyer_nom_lie(n)
        for n in fusionner_noms_lies(nom, noms_regex, noms_lev)
        if nettoyer_nom_lie(n)   # filtre les résultats vides après nettoyage
    ]
    nom_norm = normaliser(nom)

    return {
        "id":            generer_id(nom),
        "nom":           nom_norm,
        "nom_original":  nom_original(nom_norm),
        "origine_brute": texte_brut,
        "origine":       traiter_texte_origine(texte_brut, nlp),
        "noms_lies":     noms_lies,
    }


# ---------------------------------------------------------------------------
# Post-traitement : nouveaux noms + résolution id_noms_lies
# ---------------------------------------------------------------------------

def propager_liens_par_origine(resultats: list) -> None:
    """
    Propage les noms_lies entre membres d'un même groupe d'origine et attribue
    un id_groupe commun à tous les membres.

    id_groupe = generer_id du premier nom du groupe (ordre alphabétique) pour
    que l'identifiant soit stable entre runs indépendamment de l'ordre de traitement.

    Les singletons (origine unique) reçoivent leur propre id comme id_groupe.
    """
    from collections import defaultdict

    groupes: dict[str, list] = defaultdict(list)
    for entree in resultats:
        groupes[entree["origine"] or f"__solo_{entree['nom']}__"].append(entree)

    for origine, membres in groupes.items():
        # id_groupe déterministe : MD5 du nom canonique du groupe (premier par ordre alpha)
        nom_ref   = sorted(m["nom"] for m in membres)[0]
        id_groupe = generer_id(f"groupe_{nom_ref}")

        for m in membres:
            m["id_groupe"] = id_groupe

        if len(membres) < 2:
            continue

        tous_les_liens: set = set()
        for m in membres:
            tous_les_liens.update(m["noms_lies"])
            tous_les_liens.add(m["nom"])

        for m in membres:
            lies_etendus    = [n for n in tous_les_liens if n != m["nom"]]
            existants       = [n for n in m["noms_lies"] if n in set(lies_etendus)]
            complementaires = [n for n in lies_etendus if n not in existants]
            m["noms_lies"]  = (existants + complementaires)[:MAX_NOMS_LIES]


def enrichir_avec_nouveaux_noms(resultats: list) -> list:
    """
    Pipeline en 3 passes :

    Passe 1 — Index des noms existants.
    Passe 2 — Collecte des nouveaux noms découverts dans noms_lies.
    Passe 3 — Résolution id_noms_lies sur toutes les entrées.
    """
    # Passe 1 : index nom_norm -> entrée
    index: dict[str, dict] = {r["nom"]: r for r in resultats}

    # Passe 2 : collecte des nouveaux noms
    nouveaux: dict[str, dict] = {}
    for entree in resultats:
        for nom_lie in entree["noms_lies"]:
            if nom_lie in index or nom_lie in nouveaux:
                continue
            nouveaux[nom_lie] = {
                "id":            generer_id(nom_lie),
                "nom":           nom_lie,
                "nom_original":  nom_original(nom_lie),
                "origine_brute": entree.get("origine_brute", ""),
                "origine":       entree["origine"],
                "noms_lies":     [entree["nom"]],
                "id_groupe":     entree.get("id_groupe", generer_id(f"groupe_{nom_lie}")),
            }

    if nouveaux:
        log.info("%d nouveaux noms découverts et ajoutés", len(nouveaux))
        resultats = resultats + list(nouveaux.values())
        index.update(nouveaux)

    # Passe 3 : résolution id_noms_lies sur toutes les entrées
    for entree in resultats:
        entree["id_noms_lies"] = [
            index[n]["id"]
            for n in entree["noms_lies"]
            if n in index
        ]

    return resultats


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def traiter_dataset(names: list, origins_map: dict, nlp: spacy.Language) -> list:
    """
    Choix ThreadPoolExecutor :
        Levenshtein (C via rapidfuzz) et SpaCy (C en majorité) libèrent le GIL.
        Les threads partagent origins_map et nlp en lecture seule, sans verrou.
    """
    total     = len(names)
    resultats = [None] * total

    with ThreadPoolExecutor(max_workers=NB_WORKERS) as executor:
        futures = {
            executor.submit(traiter_un_nom, item, origins_map, nlp): i
            for i, item in enumerate(names)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                resultats[idx] = future.result()
            except Exception as e:
                nom = names[idx].get("name", "?")
                log.error("Erreur sur [%s] : %s", nom, e)
                resultats[idx] = {
                    "id":            generer_id(nom),
                    "nom":           normaliser(nom),
                    "nom_original":  nom_original(normaliser(nom)),
                    "origine_brute": "",
                    "origine":       "",
                    "noms_lies":     [],
                    "id_groupe":     generer_id(f"groupe_{normaliser(nom)}"),
                }
            done += 1
            if done % 1000 == 0 or done == total:
                log.info("Progression : %d / %d (%.1f%%)", done, total, 100 * done / total)

    return resultats


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def diagnostiquer(resultats: list, n_exemples: int = 10) -> None:
    avec_lies = [r for r in resultats if r["noms_lies"]]
    log.info(
        "Diagnostic : %d avec noms_lies (%.1f%%) | %d sans",
        len(avec_lies), 100 * len(avec_lies) / len(resultats),
        len(resultats) - len(avec_lies),
    )
    for r in avec_lies[:n_exemples]:
        log.info("  [%s] %s", r["nom"], r["noms_lies"])


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    for path in [INPUT_NAMES, INPUT_ORIGINS]:
        if not os.path.isfile(path):
            log.error("Fichier introuvable : %s", path)
            return

    with open(INPUT_NAMES, "r", encoding="utf-8") as f:
        names = json.load(f)
    log.info("%d noms chargés depuis %s", len(names), INPUT_NAMES)

    with open(INPUT_ORIGINS, "r", encoding="utf-8") as f:
        origins_map = json.load(f)
    log.info("%d origines chargées depuis %s", len(origins_map), INPUT_ORIGINS)

    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
        log.info("Modèle SpaCy chargé : %s", SPACY_MODEL)
    except OSError:
        log.error("Modèle '%s' introuvable. Lancer : python -m spacy download %s",
                  SPACY_MODEL, SPACY_MODEL)
        return

    resultats = traiter_dataset(names, origins_map, nlp)

    # Propagation des liens entre noms partageant la même origine
    propager_liens_par_origine(resultats)

    # Post-traitement : injection des nouveaux noms + résolution id_noms_lies
    resultats = enrichir_avec_nouveaux_noms(resultats)

    diagnostiquer(resultats)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d noms)", OUTPUT_JSON, len(resultats))


if __name__ == "__main__":
    main()