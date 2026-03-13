"""
Summarize — Phase 5
====================
Entrée  : ./data/noms_grouped.json   (produit par regroupement_noms.py)
Sorties : ./data/noms_final.json     (un objet par nom, prêt Flask)
          ./data/groupes_finals.json (un objet par groupe, prêt Flask)

Structure noms_final.json :
    {
        "id":              str,   # id MD5 du nom
        "nom":             str,   # nom normalisé
        "nom_original":    str,   # nom présentable ("Saint Oyand")
        "id_groupe_total": int,   # groupe issu de regroupement_noms.py
        "langue":          str,   # langue source détectée ("latin", "arabe"...)
        "geo":             str,   # provenance géo détectée ("bretagne"...)
        "texte_resume":    str,   # résumé extractif lisible
        "noms_lies":       list,  # noms_lies du nom lui-même (variantes directes)
        "noms_groupe":     list,  # tous les noms du groupe total
    }

Structure groupes_finals.json :
    {
        "id_groupe_total": int,
        "noms":            list,  # tous les noms du groupe
        "langue":          str,   # langue majoritaire du groupe
        "geo":             str,   # geo majoritaire du groupe
        "texte_resume":    str,   # résumé fusionné de tous les textes du groupe
    }

Résumé extractif :
    - Découpage en phrases sur les textes bruts (plus riches que les textes lemmatisés)
    - TF-IDF fit sur le corpus complet -> transform par groupe
    - Centroïde = moyenne des vecteurs de phrases normalisés L2
    - Sélection des N phrases les plus proches du centroïde
    - Déduplication cosine > DEDUP_SEUIL avant sélection
    - Réassemblage en ordre d'apparition pour la cohérence narrative

Dépendances :
    pip install scikit-learn numpy tqdm
"""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from itertools import chain

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_GROUPED  = "noms/data/2_noms_grouped.json"
OUTPUT_NOMS    = "noms/data/3_noms_final.json"
OUTPUT_GROUPES = "noms/data/3_groupes_finals.json"

MAX_PHRASES_RESUME = 4     # phrases retenues dans le résumé extractif
DEDUP_SEUIL        = 0.90  # seuil cosine de déduplication des phrases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction langue + géo (patterns regex sur texte brut)
# ---------------------------------------------------------------------------

# Patterns ordonnés par spécificité décroissante.
# On prend le premier match — évite les faux positifs ("du latin" > "latin")
PATTERNS_LANGUE = [
    (r"\bdu\s+vieux\s+(?:haut\s+)?allemand\b",  "vieux haut allemand"),
    (r"\bdu\s+vieux\s+fran[çc]ais\b",            "vieux français"),
    (r"\bdu\s+latin\b",                           "latin"),
    (r"\bdu\s+grec(?:\s+ancien)?\b",              "grec"),
    (r"\bde\s+l['\s]h[ée]breu\b",                "hébreu"),
    (r"\bde\s+l['\s]arabe\b",                     "arabe"),
    (r"\bdu\s+germanique\b",                      "germanique"),
    (r"\bdu\s+francique\b",                       "francique"),
    (r"\bdu\s+celtique?\b",                       "celte"),
    (r"\bdu\s+gaulois\b",                         "gaulois"),
    (r"\bdu\s+norrois\b",                         "norrois"),
    (r"\bdu\s+scandinave\b",                      "scandinave"),
    (r"\bdu\s+sanskrit\b",                        "sanskrit"),
    (r"\bdu\s+persan\b",                          "persan"),
    (r"\bdu\s+slave\b",                           "slave"),
    (r"\bde\s+l['\s]anglais\b",                   "anglais"),
    (r"\bdu\s+ga[ée]lique\b",                     "gaélique"),
    (r"\bd['\s]origine\s+basque\b",               "basque"),
    (r"\bd['\s]origine\s+bretonne?\b",            "breton"),
    (r"\bdu\s+gascon\b",                          "gascon"),
    (r"\bde\s+l['\s]occitan\b",                   "occitan"),
    (r"\bde\s+l['\s]alsacien\b",                  "alsacien"),
]

PATTERNS_GEO = [
    (r"\bpays\s+basque\b",    "pays basque"),
    (r"\bfranche[- ]comt[ée]\b", "franche-comté"),
    (r"\bîle[- ]de[- ]france\b", "île-de-france"),
    (r"\bnord[- ]pas[- ]de[- ]calais\b", "nord-pas-de-calais"),
    (r"\bbretagne\b",         "bretagne"),
    (r"\bnormandie\b",        "normandie"),
    (r"\balsace\b",           "alsace"),
    (r"\blorraine\b",         "lorraine"),
    (r"\bprovence\b",         "provence"),
    (r"\blanguedoc\b",        "languedoc"),
    (r"\bgascogne\b",         "gascogne"),
    (r"\bbourgogne\b",        "bourgogne"),
    (r"\bauvergne\b",         "auvergne"),
    (r"\bsavoie\b",           "savoie"),
    (r"\bdauphin[ée]\b",      "dauphiné"),
    (r"\bpoitou\b",           "poitou"),
    (r"\banjou\b",            "anjou"),
    (r"\btouraine\b",         "touraine"),
    (r"\bchampagne\b",        "champagne"),
    (r"\bpicardie\b",         "picardie"),
    (r"\bflandre\b",          "flandre"),
    (r"\bcatalogne\b",        "catalogne"),
    (r"\bjura\b",             "jura"),
    (r"\bfrance\b",           "france"),
    (r"\ballemagne\b",        "allemagne"),
    (r"\bangleterre\b",       "angleterre"),
    (r"\bitalie\b",           "italie"),
    (r"\bespagne\b",          "espagne"),
    (r"\bportugal\b",         "portugal"),
    (r"\bbelgique\b",         "belgique"),
    (r"\bsuisse\b",           "suisse"),
    (r"\bpays[- ]bas\b",      "pays-bas"),
    (r"\bscandinavie\b",      "scandinavie"),
    (r"\birlande\b",          "irlande"),
    (r"\bpologne\b",          "pologne"),
    (r"\bmaghreb\b",          "maghreb"),
    (r"\bmaroc\b",            "maroc"),
    (r"\balg[ée]rie\b",       "algérie"),
    (r"\btunisie\b",          "tunisie"),
]


def extraire_langue(texte: str) -> str:
    """Premier pattern langue matchant dans le texte, ou chaîne vide."""
    t = texte.lower()
    for pattern, label in PATTERNS_LANGUE:
        if re.search(pattern, t):
            return label
    return ""


def extraire_geo(texte: str) -> str:
    """Premier pattern géo matchant dans le texte, ou chaîne vide."""
    t = texte.lower()
    for pattern, label in PATTERNS_GEO:
        if re.search(pattern, t):
            return label
    return ""


def valeur_majoritaire(valeurs: list) -> str:
    """
    Retourne la valeur non-vide la plus fréquente dans la liste.
    Utilisé pour agréger langue et géo au niveau du groupe.
    """
    non_vides = [v for v in valeurs if v]
    if not non_vides:
        return ""
    return Counter(non_vides).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Nettoyage texte brut pour affichage
# ---------------------------------------------------------------------------

def nettoyer_pour_affichage(texte: str) -> str:
    """
    Nettoyage minimal du texte brut pour rendu HTML :
        - Normalise les espaces multiples
        - Capitalise la première lettre
        - Assure une ponctuation finale
    Ne lemmatise pas, ne supprime pas de contenu — le texte brut est
    plus lisible que le texte lemmatisé pour un utilisateur final.
    """
    if not texte or not texte.strip():
        return ""
    texte = re.sub(r"\s+", " ", texte).strip()
    texte = texte[0].upper() + texte[1:]
    if texte and texte[-1] not in ".!?":
        texte += "."
    return texte


# ---------------------------------------------------------------------------
# Résumé extractif par centroïde TF-IDF
# ---------------------------------------------------------------------------

def splitter_phrases(texte: str) -> list:
    """
    Découpe un texte en phrases sur la ponctuation forte.
    Filtre les phrases trop courtes pour être informatives (< 25 chars).
    """
    phrases = re.split(r"(?<=[.!?])\s+", texte.strip())
    return [p.strip() for p in phrases if len(p.strip()) >= 25]


def resumer_textes(textes: list, vectorizer: TfidfVectorizer) -> str:
    """
    Besoin : résumé extractif fusionné de N textes bruts pour un groupe.

    Pipeline :
        1. Découpage de tous les textes en phrases
        2. Vectorisation TF-IDF (transform sur le vectorizer déjà fitté)
        3. Déduplication : retire les phrases cosine > DEDUP_SEUIL avec une phrase déjà gardée
        4. Centroïde = moyenne L2-normalisée des vecteurs de phrases uniques
        5. Sélection des MAX_PHRASES_RESUME plus proches du centroïde
        6. Réassemblage en ordre d'apparition original (cohérence narrative)
        7. Nettoyage pour affichage

    Fallback si vectorisation échoue (textes trop courts / hors vocabulaire) :
        concaténation nettoyée des textes tronqués à 400 chars chacun.
    """
    textes_valides = [t for t in textes if t and t.strip()]
    if not textes_valides:
        return ""

    # Un seul texte : pas besoin de résumé, nettoyage direct
    if len(textes_valides) == 1:
        phrases = splitter_phrases(textes_valides[0])
        contenu = " ".join(phrases[:MAX_PHRASES_RESUME]) if phrases else textes_valides[0][:500]
        return nettoyer_pour_affichage(contenu)

    toutes_phrases = list(chain.from_iterable(splitter_phrases(t) for t in textes_valides))
    if not toutes_phrases:
        contenu = " ".join(t[:400] for t in textes_valides[:3])
        return nettoyer_pour_affichage(contenu)

    try:
        vecs      = vectorizer.transform(toutes_phrases)
        vecs_norm = normalize(vecs, norm="l2").toarray().astype(np.float32)
    except Exception:
        return nettoyer_pour_affichage(" ".join(toutes_phrases[:MAX_PHRASES_RESUME]))

    # Déduplication : O(n²) sur les phrases d'un groupe — acceptable (rarement > 50 phrases)
    gardes = [0]
    for i in range(1, len(vecs_norm)):
        sim_max = max(float(np.dot(vecs_norm[i], vecs_norm[j])) for j in gardes)
        if sim_max < DEDUP_SEUIL:
            gardes.append(i)

    phrases_uniques = [toutes_phrases[i] for i in gardes]
    embs_uniques    = vecs_norm[gardes]

    centroide = embs_uniques.mean(axis=0)
    norme     = np.linalg.norm(centroide)
    if norme > 0:
        centroide /= norme

    scores       = embs_uniques @ centroide
    top_indices  = np.argsort(scores)[::-1][:MAX_PHRASES_RESUME]
    top_ordonnes = sorted(top_indices)   # ordre d'apparition original

    contenu = " ".join(phrases_uniques[i] for i in top_ordonnes)
    return nettoyer_pour_affichage(contenu)


# ---------------------------------------------------------------------------
# Construction des sorties
# ---------------------------------------------------------------------------

def construire_sorties(noms_grouped: list, vectorizer: TfidfVectorizer) -> tuple:
    """
    Construit noms_final et groupes_finals en un seul passage.

    Logique :
        1. Regroupement des entrées par id_groupe_total
        2. Pour chaque groupe :
           a. Résumé fusionné (resumer_textes)
           b. Langue + géo majoritaires
           c. noms_groupe = liste de tous les noms du groupe
        3. Pour chaque nom : copie de ses champs + enrichissement groupe

    Retourne (noms_final, groupes_finals).
    """
    # Index nom -> entrée complète
    index_nom: dict = {d["nom"]: d for d in noms_grouped}

    # Regroupement par id_groupe_total
    groupes: dict = defaultdict(list)
    for item in noms_grouped:
        groupes[item["id_groupe_total"]].append(item)

    groupes_finals = []
    # Stockage du résumé/langue/géo par groupe pour enrichir chaque nom ensuite
    meta_groupe: dict = {}

    log.info("Construction de %d groupes...", len(groupes))

    for gid, membres in tqdm(sorted(groupes.items()), desc="Groupes", unit="groupe"):
        textes_bruts = [m.get("origine_brute", "") for m in membres]

        # Résumé fusionné
        texte_resume = resumer_textes(textes_bruts, vectorizer)

        # Langue et géo : extraction par membre puis vote majoritaire
        langues_membres = [extraire_langue(t) for t in textes_bruts]
        geos_membres    = [extraire_geo(t)    for t in textes_bruts]
        langue          = valeur_majoritaire(langues_membres)
        geo             = valeur_majoritaire(geos_membres)

        noms_du_groupe  = [m["nom"] for m in membres]

        meta_groupe[gid] = {
            "texte_resume": texte_resume,
            "langue":       langue,
            "geo":          geo,
            "noms_groupe":  noms_du_groupe,
        }

        groupes_finals.append({
            "id_groupe_total": gid,
            "noms":            sorted(noms_du_groupe),
            "langue":          langue,
            "geo":             geo,
            "texte_resume":    texte_resume,
        })

    # Construction noms_final
    log.info("Construction de %d entrées noms_final...", len(noms_grouped))
    noms_final = []
    for item in tqdm(noms_grouped, desc="Noms", unit="nom"):
        gid  = item["id_groupe_total"]
        meta = meta_groupe.get(gid, {})
        noms_final.append({
            "id":              item.get("id", ""),
            "nom":             item.get("nom", ""),
            "nom_original":    item.get("nom_original", ""),
            "id_groupe_total": gid,
            "langue":          meta.get("langue", ""),
            "geo":             meta.get("geo", ""),
            "texte_resume":    meta.get("texte_resume", ""),
            "noms_lies":       item.get("noms_lies", []),
            "noms_groupe":     meta.get("noms_groupe", []),
        })

    return noms_final, groupes_finals


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.isfile(INPUT_GROUPED):
        log.error("Fichier introuvable : %s", INPUT_GROUPED)
        return

    os.makedirs("data", exist_ok=True)

    with open(INPUT_GROUPED, "r", encoding="utf-8") as f:
        noms_grouped = json.load(f)
    log.info("%d entrées chargées depuis %s", len(noms_grouped), INPUT_GROUPED)

    # Vectorizer TF-IDF fitté sur tous les textes bruts du corpus
    # Même paramètres que regroupement_noms.py pour cohérence de l'espace vectoriel
    log.info("Entraînement TF-IDF sur textes bruts...")
    tous_textes = [d.get("origine_brute", "") or "__vide__" for d in noms_grouped]
    vectorizer  = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    vectorizer.fit(tous_textes)
    log.info("Vocabulaire TF-IDF : %d termes", len(vectorizer.vocabulary_))

    noms_final, groupes_finals = construire_sorties(noms_grouped, vectorizer)

    with open(OUTPUT_NOMS, "w", encoding="utf-8") as f:
        json.dump(noms_final, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d noms)", OUTPUT_NOMS, len(noms_final))

    with open(OUTPUT_GROUPES, "w", encoding="utf-8") as f:
        json.dump(groupes_finals, f, ensure_ascii=False, indent=2)
    log.info("Export : %s (%d groupes)", OUTPUT_GROUPES, len(groupes_finals))

    # Rapport couverture
    avec_langue = sum(1 for g in groupes_finals if g["langue"])
    avec_geo    = sum(1 for g in groupes_finals if g["geo"])
    avec_resume = sum(1 for g in groupes_finals if g["texte_resume"])
    n           = len(groupes_finals)
    log.info(
        "Couverture groupes | langue: %.1f%% | géo: %.1f%% | résumé: %.1f%%",
        100 * avec_langue / n,
        100 * avec_geo    / n,
        100 * avec_resume / n,
    )


if __name__ == "__main__":
    main()