"""
prepare_prenoms.py — Phase 1 prénoms
======================================
Entrée  : data/1_prenoms_detail_clean.json
Sortie  : data/2_prenoms_clean.json

Différences clés avec la version précédente (données lemmatisées) :
    - Les textes sources sont du français naturel (phrases complètes,
      ponctuation, majuscules). On ne lemmatise PAS : CamemBERT est
      pré-entraîné sur du français naturel et se dégrade sur des lemmes.
    - texte_brut = signification + histoire uniquement.
      caractere est exclu : descriptions de personnalité génériques,
      souvent identiques entre prénoms sans lien étymologique.
    - Nettoyage léger : suppression des URLs, balises, espaces multiples,
      conservation de la ponctuation et des majuscules.

Pour chaque prénom :
    - Normalisation du champ prenom (NFD, minuscules, tirets->espaces)
    - ID MD5 déterministe sur le prénom normalisé
    - Construction texte_brut (signification + histoire, nettoyé)
    - Extraction : langue, religion, géo, date, dérivés
    - Attribution id_groupe initial (propagation par texte identique)

Dépendances :
    pip install rapidfuzz
"""

import hashlib
import json
import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidfuzz.distance import Levenshtein as Lev

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_JSON  = "prenoms/data/1_prenoms_detail.json"
OUTPUT_JSON = "prenoms/data/2_prenoms_clean.json"
NB_WORKERS  = None  # os.cpu_count()
MAX_PRENOMS_LIES = 15


# ---------------------------------------------------------------------------
# Normalisation du prénom (champ d'index, pas du texte)
# ---------------------------------------------------------------------------

def normaliser_prenom(prenom: str) -> str:
    """
    Besoin : clé de regroupement stable et insensible aux variantes
    orthographiques mineures (accents, casse, tirets).
    Choix : NFD -> suppression diacritiques -> lower -> tirets/apostrophes
    -> espaces normalisés.
    Ne pas appliquer aux textes signification/histoire : on conserve
    les majuscules et accents pour le modèle d'embedding.
    """
    s = unicodedata.normalize("NFD", prenom)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[-'']", " ", s)
    return " ".join(s.split())


def generer_id(prenom: str) -> str:
    return hashlib.md5(normaliser_prenom(prenom).encode()).hexdigest()[:8]


def prenom_original(prenom_norm: str) -> str:
    return " ".join(w.capitalize() for w in prenom_norm.split())


# ---------------------------------------------------------------------------
# Nettoyage du texte naturel (signification / histoire)
# ---------------------------------------------------------------------------

# Balises HTML résiduelles éventuelles
_RE_BALISE    = re.compile(r"<[^>]+>")
# URLs
_RE_URL       = re.compile(r"https?://\S+")
# Espaces multiples
_RE_ESPACES   = re.compile(r" {2,}")


def nettoyer_texte(texte: str) -> str:
    """
    Besoin : retirer le bruit sans dégrader le signal sémantique.
    Choix : suppression URLs + balises HTML + normalisation espaces.
    On conserve volontairement la ponctuation, les majuscules et les
    accents : CamemBERT les utilise pour segmenter et contextualiser.
    Pas de lemmatisation : dégraderait les embeddings sur du français naturel.
    """
    if not texte:
        return ""
    t = _RE_BALISE.sub(" ", texte)
    t = _RE_URL.sub(" ", t)
    t = _RE_ESPACES.sub(" ", t)
    return t.strip()


def construire_texte_brut(item: dict) -> str:
    """
    Besoin : produire la représentation textuelle pour l'embedding.
    Champs retenus : signification + histoire.
    caractere exclu : personnalité générique, souvent partagée entre
    prénoms sans lien étymologique -> pollue la similarité cosine.
    histoire répétée deux fois pour amplifier le signal étymologique
    dans l'espace d'embedding sans modifier le modèle.
    Séparateur " | " conservé pour lisibilité dans les exports.
    """
    # sig  = nettoyer_texte(item.get("signification", ""))
    hist = nettoyer_texte(item.get("histoire", ""))
    prov = nettoyer_texte(item.get("provenance", ""))
    ety = nettoyer_texte(item.get("etymologie", ""))

    parties = []
    # if sig:
    #     parties.append(sig)
    
    if hist:
        # Répétition de histoire : poids étymologique renforcé
        parties.append(hist)
        parties.append(hist)
    if prov:
        parties.append(prov)
    if ety:
        parties.append(ety)
    return " | ".join(parties)


# ---------------------------------------------------------------------------
# Listes fermées — langue, religion, géo
# ---------------------------------------------------------------------------

PATTERNS_LANGUE = [
    (r"\bvieux\s+(?:haut\s+)?allemand\b",  "vieux haut allemand"),
    (r"\bvieux\s+fran[çc]ais\b",            "vieux français"),
    (r"\bh[ée]breu\b",                      "hébreu"),
    (r"\barabe\b",                          "arabe"),
    (r"\blatin\b",                          "latin"),
    (r"\bgrec(?:\s+ancien)?\b",             "grec"),
    (r"\bgermanique\b",                     "germanique"),
    (r"\bfrancique\b",                      "francique"),
    (r"\bceltique?\b",                      "celte"),
    (r"\bgaulois\b",                        "gaulois"),
    (r"\bnorrois\b",                        "norrois"),
    (r"\bscandinave\b",                     "scandinave"),
    (r"\bsanskrit\b",                       "sanskrit"),
    (r"\bpersan\b",                         "persan"),
    (r"\bslave\b",                          "slave"),
    (r"\banglais\b",                        "anglais"),
    (r"\bga[ée]lique\b",                    "gaélique"),
    (r"\bbasque\b",                         "basque"),
    (r"\bbreton\b",                         "breton"),
    (r"\bgascon\b",                         "gascon"),
    (r"\boccitan\b",                        "occitan"),
    (r"\balsacien\b",                       "alsacien"),
    (r"\bturque?\b",                        "turc"),
    (r"\bpolin[ée]sien\b",                  "polynésien"),
    (r"\bjaponais\b",                       "japonais"),
    (r"\bchinois\b",                        "chinois"),
]

PATTERNS_RELIGION = [
    (r"\bsaint\b|\bsainte\b",               "chrétien"),
    (r"\bbible\b|\bbiblique\b",             "chrétien"),
    (r"\bap[ôo]tre\b",                      "chrétien"),
    (r"\b[ée]vangile\b|\b[ée]vang[ée]lique\b", "chrétien"),
    (r"\bpape\b|\b[ée]v[êe]que\b|\babb[ée]\b", "chrétien"),
    (r"\bchristianisme\b|\bchr[ée]tien\b",  "chrétien"),
    (r"\bislam\b|\bmusulman\b|\bcoran\b",   "musulman"),
    (r"\bproph[eè]te\b",                    "musulman"),
    (r"\bjuif\b|\bh[ée]breu\b|\btorah\b",  "juif"),
    (r"\bsynagogue\b|\brabin\b",            "juif"),
    (r"\bdruidique\b|\bceltique\b",         "païen"),
    (r"\bmythologie\b",                     "mythologique"),
    (r"\bhindu\b|\bhindouisme\b",           "hindou"),
    (r"\bbouddhiste?\b",                    "bouddhiste"),
]

PATTERNS_GEO = [
    (r"\bpays\s+basque\b",      "pays basque"),
    (r"\bfranche[- ]comt[ée]\b","franche-comté"),
    (r"\bbretagne\b",           "bretagne"),
    (r"\bnormandie\b",          "normandie"),
    (r"\balsace\b",             "alsace"),
    (r"\blorraine\b",           "lorraine"),
    (r"\bprovence\b",           "provence"),
    (r"\blanguedoc\b",          "languedoc"),
    (r"\bgascogne\b",           "gascogne"),
    (r"\bbourgogne\b",          "bourgogne"),
    (r"\bauvergne\b",           "auvergne"),
    (r"\bsavoie\b",             "savoie"),
    (r"\bjura\b",               "jura"),
    (r"\bfrance\b|\bfran[çc]ais\b", "france"),
    (r"\ballemagne\b|\ballemand\b",  "allemagne"),
    (r"\bangleterre\b|\banglais\b",  "angleterre"),
    (r"\bitalie\b|\bitalien\b",      "italie"),
    (r"\bespagne\b|\bespagnol\b",    "espagne"),
    (r"\bportugal\b|\bportugais\b",  "portugal"),
    (r"\bbelgique\b",           "belgique"),
    (r"\bsuisse\b",             "suisse"),
    (r"\bscandinavie\b",        "scandinavie"),
    (r"\birlande\b|\birlandais\b",   "irlande"),
    (r"\bpologne\b|\bpolonais\b",    "pologne"),
    (r"\bmaghreb\b",            "maghreb"),
    (r"\b[ée]cosse\b|\b[ée]cossais\b", "écosse"),
    (r"\bpays[- ]bas\b",        "pays-bas"),
    (r"\bgr[èe]ce\b|\bgrec\b",  "grèce"),
    (r"\bturquie\b|\bturc\b",   "turquie"),
    (r"\bisra[eë]l\b",          "israël"),
    (r"\bar[aâ]bie\b",          "arabie"),
    (r"\bafrique\b|\bafricain\b",    "afrique"),
    (r"\b[ée]gypte\b|\b[ée]gyptien\b", "égypte"),
    (r"\bhexagone\b",           "france"),
]

PATTERNS_DATE = [
    (r"\b(X{0,3}(?:IX|IV|V?I{0,3})(?:e?|[eè]me))\s*si[èe]cle\b", "siecle"),
    (r"\b(1[0-9]{3}|20[0-2][0-9])\b",                              "annee"),
]

SIECLE_MAP = {
    "ier":1,"ie":1,"i":1,"iie":2,"ii":2,"iiie":3,"iii":3,
    "ive":4,"iv":4,"ve":5,"v":5,"vie":6,"vi":6,
    "viie":7,"vii":7,"viiie":8,"viii":8,"ixe":9,"ix":9,
    "xe":10,"x":10,"xie":11,"xi":11,"xiie":12,"xii":12,
    "xiiie":13,"xiii":13,"xive":14,"xiv":14,"xve":15,"xv":15,
    "xvie":16,"xvi":16,"xviie":17,"xvii":17,"xviiie":18,"xviii":18,
    "xixe":19,"xix":19,"xxe":20,"xx":20,"xxie":21,"xxi":21,
    # Formes avec accents (IVème, XIXème)
    "ivème":4,"vème":5,"viième":7,"xème":10,"xième":11,
    "xiiième":13,"xivème":14,"xvème":15,"xvième":16,
    "xviième":17,"xviiième":18,"xixème":19,"xxème":20,
}


def extraire_langue(texte: str) -> str:
    t = texte.lower()
    for pattern, label in PATTERNS_LANGUE:
        if re.search(pattern, t):
            return label
    return ""


def extraire_religion(texte: str) -> str:
    t = texte.lower()
    for pattern, label in PATTERNS_RELIGION:
        if re.search(pattern, t):
            return label
    return ""


def extraire_geo(texte: str) -> str:
    t = texte.lower()
    for pattern, label in PATTERNS_GEO:
        if re.search(pattern, t):
            return label
    return ""


def extraire_date(texte: str) -> dict:
    """Retourne {"label": str, "valeur": int|None}."""
    if not texte:
        return {"label": "", "valeur": None}
    t = texte.lower()
    m = re.search(PATTERNS_DATE[0][0], t, re.IGNORECASE)
    if m:
        code   = m.group(1).lower().replace(" ", "").replace("è", "e").replace("é", "e")
        siecle = SIECLE_MAP.get(code)
        if siecle:
            return {
                "label":  f"{m.group(1).upper()} siècle",
                "valeur": (siecle - 1) * 100 + 50,
            }
    m = re.search(PATTERNS_DATE[1][0], t)
    if m:
        annee = int(m.group(1))
        return {"label": str(annee), "valeur": annee}
    return {"label": "", "valeur": None}


# ---------------------------------------------------------------------------
# Extraction de prénoms liés — sections étiquetées (Variantes / Dérivés)
# ---------------------------------------------------------------------------

# Patterns de sections étiquetées — ordre décroissant de spécificité
_PATTERNS_VARIANTES = [
    r"variantes?\s+ou\s+formes?\s+voisines?\s*:",
    r"formes?\s+voisines?\s+ou\s+variantes?\s*:",
    r"variantes?\s*:",
    r"formes?\s+voisines?\s*:",
    r"formes?\s+apparentées?\s*:",
]
_PATTERNS_DERIVES = [
    r"dérivés?\s+ou\s+formes?\s+dérivées?\s*:",
    r"formes?\s+dérivées?\s+ou\s+dérivés?\s*:",
    r"dérivés?\s*:",
    r"formes?\s+dérivées?\s*:",
    r"diminutifs?\s*:",
]

_SEP_PRENOMS = re.compile(r"[,;]+")
_FIN_SECTION = re.compile(
    r"(?:variantes?|dérivés?|formes?|diminutifs?|hypocoristiques?"
    r"|fréquence|étymologie|origine|histoire|provenance|note\s*:)",
    re.IGNORECASE,
)


def _extraire_liste_apres_label(texte: str, patterns: list) -> list:
    """
    Besoin : extraire la liste de prénoms qui suit un label structurant
    ("Variantes :", "Dérivés :", etc.) dans le texte libre.
    Choix : recherche du label dans la version lowercasée, extraction du
    segment jusqu'au prochain label ou à la première phrase suivante,
    suppression du contenu entre parenthèses avant le split.
    """
    texte_lower = texte.lower()
    for pattern in patterns:
        match = re.search(pattern, texte_lower, re.IGNORECASE)
        if not match:
            continue
        segment = texte[match.end():]
        fin = re.search(r"\.\s+[A-Z]|\.\s*$", segment)
        if fin:
            segment = segment[:fin.start()]
        fin_lbl = _FIN_SECTION.search(segment)
        if fin_lbl:
            segment = segment[:fin_lbl.start()]
        # Contenu entre parenthèses supprimé : "(Picardie)", "(17)" -> ""
        segment = re.sub(r"\([^)]*\)", "", segment)
        noms = [t.strip().strip(".") for t in _SEP_PRENOMS.split(segment)]
        noms = [n for n in noms if len(n) >= 2 and not n.isdigit()]
        if noms:
            return noms
    return []


def extraire_prenoms_regex(texte: str) -> list:
    """Variantes + dérivés étiquetés, fusionnés sans doublon."""
    vus, fusion = set(), []
    for n in (_extraire_liste_apres_label(texte, _PATTERNS_VARIANTES) +
              _extraire_liste_apres_label(texte, _PATTERNS_DERIVES)):
        cle = n.lower().strip()
        if cle not in vus:
            fusion.append(n)
            vus.add(cle)
    return fusion


# ---------------------------------------------------------------------------
# Extraction de prénoms liés — candidats Levenshtein depuis le texte brut
# ---------------------------------------------------------------------------

# Mot simple capitalisé : "Marie", "Pierre"
_RE_MOT_MAJUSCULE_P = re.compile(r"\b([A-ZÀ-Ü][a-zà-ü]{2,})\b")

# Prénom composé avec tiret : "Jean-Marie", "Anne-Sophie", "Marie-Pier"
# Chaque composante commence par une majuscule, séparées par un ou plusieurs tirets.
# Pas de re.IGNORECASE : "jean-marie" ne matche pas, seul "Jean-Marie" oui.
_RE_PRENOM_COMPOSE = re.compile(
    r"\b([A-ZÀ-Ü][a-zà-ü]{1,}(?:-[A-ZÀ-Ü][a-zà-ü]{1,})+)\b"
)

# Tokens balisés typographiquement (backticks, italique) en minuscule
_RE_BALISE_P = re.compile(r"[`*_]([A-Za-zÀ-ÿ\-]{2,})[`*_]")

# Énumérations contextuelles après un signal de dérivation dans le corps du texte.
# Groupe capturant : séquence stricte de tokens sans espace séparés par , ; et ou puis
# -> "Ludwig, Ludovic, Clovis, etc." capturé
# -> "des indigents" : pas de majuscule après le signal, non capturé
# -> "Aelis et Alis sont de nouvelles tendances" : arrêt avant "sont" (non séparateur)
# -> "entre Adalis, Alis, puis Alix" : capturé via signal "entre"
_RE_ENUM_SIGNAL = re.compile(
    r"(?:a\s+donn[eé](?:\s+aussi|\s+[eé]galement)?|"
    r"se\s+d[eé]cline(?:\s+aussi|\s+[eé]galement)?\s+en|"
    r"tel(?:s|les?)?\s+que|"
    r"(?:formes?\s+)?(?:anglaise|espagnole|italienne|allemande|latine|"
    r"bretonne|occitane|catalane|portugaise|polonaise|russe|scandinave)s?\s*:|"
    r"comme(?=\s+[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ])|"
    r"notamment(?=\s+[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ])|"
    r"entre(?=\s+[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ])"       # "entre Adalis, Alis, puis Alix"
    r")\s+"
    r"((?:[A-ZÀ-Üa-zà-ü][A-Za-zÀ-ÿ\-]{1,19})"
    r"(?:\s*(?:[,;]\s*(?:(?:et|ou|and|or|puis)\s+)?|(?:\s+(?:et|ou|and|or|puis)\s+))(?!(?:et|ou|and|or|puis)\b)[A-ZÀ-Üa-zà-ü][A-Za-zÀ-ÿ\-]{1,19})*)",
    re.IGNORECASE,
)
_SEP_ENUM = re.compile(r"[,;]+|\s+(?:et|ou|and|or|puis)\s+")

# Mots grammaticaux et termes parasites à exclure de la liste de prénoms
_MOTS_GRAM = frozenset({
    "des", "les", "une", "un", "du", "de", "la", "le", "aux", "au",
    "ses", "ces", "mes", "tes", "nos", "vos", "leur", "leurs",
    "ce", "cet", "cette", "mon", "ton", "son",
    "par", "pour", "sur", "sous", "dans", "avec", "sans", "entre",
    "qui", "que", "quoi", "dont", "ou", "et", "ni", "mais", "puis",
    "plus", "tres", "aussi", "ainsi", "notamment", "comme",
    "indigents", "indigentes", "personnes", "gens", "autres",
    "etc",   # filtre le token "etc" capturé en fin de liste
})


def extraire_prenoms_enum(texte: str) -> list:
    """
    Besoin : capturer les listes de prénoms apparentés citées dans le corps
    du texte après un signal de dérivation non structuré (sans label de section).

    Cas couverts :
        "qui a donné aussi Ludwig, Ludovic, Clovis, etc."
        "se décline aussi en Alicia"
        "comme Aelis et Alis"
        "forme italienne : Lucia, Lucie"

    Filtres appliqués sur chaque token extrait :
        1. Pas d'espace interne — un prénom est un mot ou un composé tiret,
           jamais "des indigents" ou "nouvelles tendances".
        2. Longueur 2–20 caractères.
        3. Pas un mot grammatical (_MOTS_GRAM).
        4. Commence par une lettre.
    """
    resultats = []
    vus = set()
    for m in _RE_ENUM_SIGNAL.finditer(texte):
        segment = re.sub(r"\([^)]*\)", "", m.group(1))
        for token in _SEP_ENUM.split(segment):
            token = token.strip().strip(".")
            # Filtre 1 : pas d'espace interne (élimine "des indigents", "nouvelles tendances")
            if " " in token:
                continue
            # Filtre 2 : longueur
            if len(token) < 2 or len(token) > 20:
                continue
            # Filtre 3 : commence par une lettre
            if not re.match(r"[A-Za-zÀ-ÿ]", token):
                continue
            # Filtre 4 : pas un mot grammatical (comparaison en minuscules sans accents)
            norm = normaliser_prenom(token)
            if not norm or norm in _MOTS_GRAM or norm in vus:
                continue
            resultats.append(token)
            vus.add(norm)
    return resultats


# Termes géographiques et communs à exclure de la détection Levenshtein
_EXCLUSIONS_LEV = frozenset({
    "france", "europe", "afrique", "asie", "egypte", "rome", "paris",
    "saint", "sainte", "dieu", "bible", "coran", "islam", "christianisme",
    "janvier", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "aout", "septembre", "octobre", "novembre", "decembre",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "france", "bretagne", "alsace", "normandie", "provence",
    "nord", "sud", "est", "ouest",
})


def extraire_candidats_prenoms(texte_brut: str) -> list:
    """
    Collecte les candidats pour le filtre Levenshtein depuis trois sources.

    1. Prénoms composés avec tiret (priorité haute) :
       "...également porté sous la forme Jean-Marie..." -> ["Jean-Marie"]
       Traité en premier pour que "Jean-Marie" soit enregistré avant
       que "Jean" et "Marie" soient ajoutés séparément par la source 2.

    2. Mots simples capitalisés :
       "...forme féminine Marie..." -> ["Marie"]

    3. Tokens balisés typographiquement :
       "...issu du terme *iohannes*..." -> ["iohannes"]

    Les énumérations contextuelles ("a donné aussi X, Y, Z") contournent
    ce filtre et sont traitées directement par extraire_prenoms_enum.
    """
    candidats = []
    vus_norm  = set()

    def ajouter(c: str) -> None:
        n = normaliser_prenom(c)
        if n and n not in vus_norm:
            candidats.append(c)
            vus_norm.add(n)

    # Priorité 1 : composés avec tiret
    for m in _RE_PRENOM_COMPOSE.findall(texte_brut):
        ajouter(m)
    # Priorité 2 : mots simples capitalisés
    for m in _RE_MOT_MAJUSCULE_P.findall(texte_brut):
        ajouter(m)
    # Priorité 3 : balisés
    for m in _RE_BALISE_P.findall(texte_brut):
        ajouter(m)

    return candidats


def seuil_lev_prenom(longueur: int) -> int:
    """
    Seuil Levenshtein adaptatif selon la longueur du prénom normalisé.
    Même barème que prepare_noms, calibré sur des cas réels :
        <= 4 : seuil 1  (Luc/Luca, Eva/Eve)
        <= 7 : seuil 2  (Pierre/Piere, Marie/Mario)
        >  7 : seuil 3  (Alexandre/Alexandr)
    Les prénoms composés normalisés (ex. "jean marie" = 9 chars) tombent
    dans la troisième tranche, ce qui est intentionnel.
    """
    if longueur <= 4:
        return 1
    if longueur <= 7:
        return 2
    return 3


def extraire_prenoms_lies_lev(prenom: str, texte_brut: str) -> list:
    """
    Besoin : trouver dans le texte les candidats orthographiquement proches
    du prénom source, pour capturer variantes non étiquetées.

    Logique :
        1. Collecter candidats via extraire_candidats_prenoms.
        2. Normaliser + filtrer exclusions géo/communes.
        3. Distance Levenshtein sur formes normalisées (tirets -> espaces).
        4. Garder si dist <= seuil adaptatif.
    """
    prenom_norm = normaliser_prenom(prenom)
    seuil       = seuil_lev_prenom(len(prenom_norm))
    vus         = {prenom_norm}
    prenoms_lies = []

    for candidat in extraire_candidats_prenoms(texte_brut):
        cand_norm = normaliser_prenom(candidat)
        if cand_norm in vus or cand_norm in _EXCLUSIONS_LEV:
            continue
        dist = Lev.distance(prenom_norm, cand_norm, score_cutoff=seuil)
        if dist is not None and dist <= seuil:
            prenoms_lies.append(candidat)
            vus.add(cand_norm)

    return prenoms_lies


# ---------------------------------------------------------------------------
# Fusion et nettoyage des prénoms liés
# ---------------------------------------------------------------------------

def nettoyer_prenom_lie(prenom: str) -> str:
    """
    Besoin : normaliser un prénom lié extrait (regex ou Levenshtein)
    pour qu'il soit homogène avec le champ 'prenom' du JSON.
    Contenu entre parenthèses supprimé en filet de sécurité
    (cas où un candidat Lev contiendrait une parenthèse résiduelle).
    """
    prenom = re.sub(r"\([^)]*\)", "", prenom)
    prenom = re.sub(r"[\d]", "", prenom)
    prenom = re.sub(r"[^\w\s\-]", "", prenom)
    return normaliser_prenom(prenom.strip())


def fusionner_prenoms_lies(prenom_source: str, *sources) -> list:
    """Union ordonnée, sans le prénom source, sans doublon, tronquée à MAX_PRENOMS_LIES."""
    source_norm = normaliser_prenom(prenom_source)
    vus, liste  = {source_norm}, []
    for source in sources:
        for p in source:
            cle = normaliser_prenom(p)
            if cle not in vus:
                liste.append(p)
                vus.add(cle)
    return liste[:MAX_PRENOMS_LIES]


# ---------------------------------------------------------------------------
# Post-traitement : résolution id_prenoms_lies + injection nouveaux prénoms
# ---------------------------------------------------------------------------

def enrichir_avec_nouveaux_prenoms(resultats: list) -> list:
    """
    Besoin : résoudre les prénoms_lies en identifiants et injecter dans
    le dataset les prénoms découverts qui n'étaient pas dans la source.

    Pipeline en 3 passes — même logique que prepare_noms.enrichir_avec_nouveaux_noms :

    Passe 1 : index prenom_norm -> entrée (O(n)).
    Passe 2 : collecte des prénoms découverts absents de l'index.
              Chaque nouveau prénom hérite de l'id_groupe de son découvreur
              et pointe vers lui dans ses propres prenoms_lies.
    Passe 3 : résolution id_prenoms_lies sur toutes les entrées.
    """
    # Passe 1
    index: dict[str, dict] = {r["prenom"]: r for r in resultats}

    # Passe 2
    nouveaux: dict[str, dict] = {}
    for entree in resultats:
        for p in entree.get("prenoms_lies", []):
            if p in index or p in nouveaux:
                continue
            nouveaux[p] = {
                "id":               generer_id(p),
                "prenom":           p,
                "prenom_original":  prenom_original(p),
                "sexe":             "",
                "url":              "",
                "signification":    None,
                "histoire":         None,
                "etymologie":       None,
                "provenance":       None,
                "texte_brut":       entree.get("texte_brut", ""),
                "langue":           entree.get("langue", ""),
                "religion":         entree.get("religion", ""),
                "geo":              entree.get("geo", ""),
                "date":             entree.get("date", {"label": "", "valeur": None}),
                "prenoms_lies":     [entree["prenom"]],
                "id_groupe":        entree.get("id_groupe", generer_id(f"solo_{p}")),
            }

    if nouveaux:
        log.info("%d nouveaux prenoms decouverts et ajoutes", len(nouveaux))
        resultats = resultats + list(nouveaux.values())
        index.update(nouveaux)

    # Passe 3
    for entree in resultats:
        entree["id_prenoms_lies"] = [
            index[p]["id"]
            for p in entree.get("prenoms_lies", [])
            if p in index
        ]

    return resultats


def diagnostiquer_lies(resultats: list, n_exemples: int = 10) -> None:
    avec_lies = [r for r in resultats if r.get("prenoms_lies")]
    log.info(
        "Diagnostic prenoms_lies : %d avec liens (%.1f%%) | %d sans",
        len(avec_lies), 100 * len(avec_lies) / max(len(resultats), 1),
        len(resultats) - len(avec_lies),
    )
    for r in avec_lies[:n_exemples]:
        log.info("  [%s] -> %s", r["prenom"], r["prenoms_lies"])


# ---------------------------------------------------------------------------
# Traitement unitaire
# ---------------------------------------------------------------------------

def traiter_un_prenom(item: dict) -> dict:
    """
    Besoin : transformer une entrée brute en entrée normalisée pour la phase 2.
    texte_brut = histoire + provenance + etymologie nettoyés (sans lemmatisation).
    Les extractions langue/religion/géo/date opèrent sur le texte complet.

    prenoms_lies : union de trois sources sur les champs histoire + etymologie + provenance.
        Source 1 (regex sections) : "Variantes :", "Dérivés :", "Diminutifs :"
        Source 2 (enum corps)     : "a donné aussi X, Y, Z", "notamment X, Y"
                                    — contourne Levenshtein, capture les prénoms
                                    étymologiquement liés mais orthographiquement éloignés
                                    (Ludwig, Clovis, Luigi pour Ludovica).
        Source 3 (Levenshtein)    : candidats orthographiquement proches du prénom,
                                    en privilégiant les prénoms composés (Jean-Marie).
    La résolution id_prenoms_lies se fait en post-traitement dans main(),
    une fois que tous les IDs sont connus.
    """
    prenom_raw  = item.get("prenom", "").strip()
    prenom_norm = normaliser_prenom(prenom_raw)

    # Texte complet pour les extractions de métadonnées
    texte_complet = " ".join(filter(None, [
        nettoyer_texte(item.get("signification", "")),
        nettoyer_texte(item.get("histoire", "")),
        nettoyer_texte(item.get("etymologie", "")),
        nettoyer_texte(item.get("provenance", "")),
    ]))

    # Texte pour l'embedding : histoire + provenance + etymologie
    texte_brut = construire_texte_brut(item)

    # Texte source pour les prénoms liés : histoire + etymologie + provenance
    # (signification exclue : trop générique, génère du bruit)
    texte_liens = " ".join(filter(None, [
        nettoyer_texte(item.get("histoire", "")),
        nettoyer_texte(item.get("etymologie", "")),
        nettoyer_texte(item.get("provenance", "")),
    ]))

    prenoms_regex = extraire_prenoms_regex(texte_liens)
    prenoms_enum  = extraire_prenoms_enum(texte_liens)
    prenoms_lev   = extraire_prenoms_lies_lev(prenom_raw, texte_liens)
    # Ordre de priorité : regex (sections étiquetées) > enum (corps de phrase) > Lev
    prenoms_lies  = [
        nettoyer_prenom_lie(p)
        for p in fusionner_prenoms_lies(prenom_raw, prenoms_regex, prenoms_enum, prenoms_lev)
        if nettoyer_prenom_lie(p)
    ]

    return {
        "id":               generer_id(prenom_raw),
        "prenom":           prenom_norm,
        "prenom_original":  prenom_original(prenom_norm),
        "sexe":             item.get("sexe", ""),
        "url":              item.get("url", ""),
        "signification":    item.get("signification"),
        "histoire":         item.get("histoire"),
        "etymologie":       item.get("etymologie"),
        "provenance":       item.get("provenance"),
        "texte_brut":       texte_brut,
        "langue":           extraire_langue(texte_complet),
        "religion":         extraire_religion(texte_complet),
        "geo":              extraire_geo(texte_complet),
        "date":             extraire_date(texte_complet),
        "prenoms_lies":     prenoms_lies,
    }


# ---------------------------------------------------------------------------
# Propagation id_groupe initial (texte identique)
# ---------------------------------------------------------------------------

def propager_groupes(resultats: list) -> None:
    """
    Besoin : pré-regrouper les prénoms partageant exactement le même texte_brut,
    typiquement les paires masculin/féminin issues de la même fiche source.
    Ces groupes initiaux sont injectés dans l'Union-Find de la phase 2.
    Modification en place.
    """
    texte_to_groupe: dict = {}
    for r in resultats:
        t = r["texte_brut"]
        if not t:
            # Prénoms sans texte : groupe individuel pour éviter les faux positifs
            r["id_groupe"] = generer_id(f"solo_{r['prenom']}")
            continue
        if t not in texte_to_groupe:
            texte_to_groupe[t] = generer_id(f"grp_{r['prenom']}")
        r["id_groupe"] = texte_to_groupe[t]

    n_groupes  = len(set(r["id_groupe"] for r in resultats))
    n_non_solo = sum(
        1 for r in resultats
        if sum(1 for s in resultats if s["id_groupe"] == r["id_groupe"]) > 1
    )
    log.info(
        "Groupes initiaux : %d | prenoms non-solo : %d / %d",
        n_groupes, n_non_solo, len(resultats),
    )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.isfile(INPUT_JSON):
        log.error("Fichier introuvable : %s", INPUT_JSON)
        return

    os.makedirs("data", exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info("%d prenoms charges", len(data))

    resultats  = [None] * len(data)
    nb_workers = NB_WORKERS or os.cpu_count() or 1

    with ThreadPoolExecutor(max_workers=nb_workers) as executor:
        futures = {
            executor.submit(traiter_un_prenom, item): i
            for i, item in enumerate(data)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                resultats[idx] = future.result()
            except Exception as e:
                nom = data[idx].get("prenom", "?")
                log.error("Erreur sur '%s' : %s", nom, e)
                nom_norm = normaliser_prenom(nom)
                resultats[idx] = {
                    "id":              generer_id(nom),
                    "prenom":          nom_norm,
                    "prenom_original": prenom_original(nom_norm),
                    "sexe":            data[idx].get("sexe", ""),
                    "url":             data[idx].get("url", ""),
                    "signification":   None,
                    "histoire":        None,
                    "etymologie":      None,
                    "provenance":      None,
                    "texte_brut":      "",
                    "langue":          "",
                    "religion":        "",
                    "geo":             "",
                    "date":            {"label": "", "valeur": None},
                    "prenoms_lies":    [],
                    "id_prenoms_lies": [],
                    "id_groupe":       generer_id(f"solo_{nom_norm}"),
                }

    propager_groupes(resultats)

    # Post-traitement : injection des prénoms découverts + résolution id_prenoms_lies
    resultats = enrichir_avec_nouveaux_prenoms(resultats)

    diagnostiquer_lies(resultats)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d prenoms)", OUTPUT_JSON, len(resultats))

    # Rapport de couverture des métadonnées
    for champ in ("langue", "religion", "geo"):
        n = sum(1 for r in resultats if r.get(champ))
        log.info("Couverture %s : %.1f%%", champ, 100 * n / len(resultats))
    n_date  = sum(1 for r in resultats if r.get("date", {}).get("valeur"))
    n_vide  = sum(1 for r in resultats if not r.get("texte_brut"))
    n_lies  = sum(1 for r in resultats if r.get("prenoms_lies"))
    log.info("Couverture date         : %.1f%%",     100 * n_date / len(resultats))
    log.info("Prenoms sans texte      : %d / %d",    n_vide, len(resultats))
    log.info("Prenoms avec lies       : %d / %d",    n_lies, len(resultats))


if __name__ == "__main__":
    main()