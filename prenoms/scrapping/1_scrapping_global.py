"""
Scraper Phase 1 - prenoms.com (version async)
Objectif  : collecter tous les prénoms par lettre -> un CSV par lettre dans /data
Dépendances :
    pip install httpx beautifulsoup4 unidecode pandas

Choix technique : httpx + asyncio
    - requests est synchrone : chaque requête bloque le thread. Sur 26 lettres x ~60 pages,
      le gain async est significatif (x3 à x5 en pratique sur du I/O réseau pur).
    - Scrapy aurait été plus rigide à piloter interactivement (event loop Twisted isolé).
    - httpx expose la même API que requests, migration quasi sans friction.
    - asyncio.Semaphore contrôle la concurrence pour ne pas surcharger le serveur.
"""

import asyncio
import math
import os
import random
import re
import logging
from typing import Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL         = "https://www.prenoms.com/recherche/prenoms"
DATA_DIR         = "data"
PRENOMS_PAR_PAGE = 30

# Concurrence max entre lettres traitées en parallèle.
# Valeur conservatrice : au-dela de 3-4, risque de ban IP.
MAX_CONCURRENT_LETTRES = 3

# Concurrence max entre pages d'une même lettre
MAX_CONCURRENT_PAGES = 4

DELAY_MIN = 0.8
DELAY_MAX = 2.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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

def normaliser_prenom(raw: str) -> str:
    """
    Besoin : extraire le prénom normalisé depuis l'attribut title de la balise <a>.
    Choix  : split sur le préfixe fixe, puis unidecode + lower. Pas de regex complexe.
    Usage  : normaliser_prenom('Tout sur le prénom Amélie') -> 'amelie'
    """
    prefixe = "Tout sur le prénom "
    if prefixe in raw:
        raw = raw.split(prefixe, 1)[1]
    normalise = unidecode(raw).lower().strip()
    normalise = re.sub(r"[^a-z\-]", "", normalise)
    return normalise


def extraire_sexe(url: str) -> str:
    if "prenom-fille" in url:
        return "fille"
    elif "prenom-garcon" in url:
        return "garcon"
    return "inconnu"


def csv_path(lettre: str) -> str:
    return os.path.join(DATA_DIR, f"prenoms_{lettre}.csv")


def lettre_deja_scrapee(lettre: str) -> bool:
    """
    Besoin : permettre la reprise sans re-scraper les lettres déjà traitées.
    Choix  : vérification de l'existence du CSV de la lettre dans /data.
    """
    return os.path.isfile(csv_path(lettre))


# ---------------------------------------------------------------------------
# Parsing HTML
# ---------------------------------------------------------------------------

def parser_nombre_prenoms(soup: BeautifulSoup) -> int:
    """
    Besoin : extraire le nombre total de prénoms depuis le subtitle de la page.
    Choix  : class CSS 'subtitle leftcolor mb-3', on extrait le premier entier trouvé.
             Plus fiable que de compter les page-link (pagination parfois incohérente).
    Usage  : parser_nombre_prenoms(soup) -> 1741
             -> nombre de pages = ceil(1741 / 30)
    """
    el = soup.select_one("p.subtitle.leftcolor.mb-3")
    if el:
        texte = el.get_text(strip=True)
        match = re.search(r"(\d[\d\s]*)", texte)
        if match:
            return int(match.group(1).replace(" ", ""))
    log.warning("Impossible de lire le nombre de prénoms dans le subtitle.")
    return 0


def parser_prenoms_page(soup: BeautifulSoup) -> list[dict]:
    """
    Besoin : extraire tous les prénoms d'une page de résultats.
    Choix  : sélecteur CSS 'a.prenom-title', stable selon la structure fournie.
    Usage  : retourne [{prenom, url, sexe}]
    """
    resultats = []
    for lien in soup.select("a.prenom-title"):
        url   = lien.get("href", "")
        title = lien.get("title", "")
        if not url or not title:
            continue
        resultats.append({
            "prenom": normaliser_prenom(title),
            "url":    url,
            "sexe":   extraire_sexe(url),
        })
    return resultats


# ---------------------------------------------------------------------------
# Requêtes async
# ---------------------------------------------------------------------------

async def fetch_page(
    client: httpx.AsyncClient,
    lettre: str,
    page: int,
    sem: asyncio.Semaphore,
) -> Optional[BeautifulSoup]:
    """
    Besoin : requête HTTP async avec retry exponentiel et délai aléatoire.
    Choix  : httpx.AsyncClient réutilisé (keep-alive, pool de connexions).
             Semaphore injecté pour limiter la concurrence sans coupling fort.
    Usage  : await fetch_page(client, 'A', 2, sem)
    """
    params = {"firstletter": lettre, "page": page}
    async with sem:
        await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        for attempt in range(1, 4):
            try:
                r = await client.get(BASE_URL, params=params, timeout=15)
                r.raise_for_status()
                return BeautifulSoup(r.text, "html.parser")
            except httpx.HTTPStatusError as e:
                log.warning("[%s p%d] HTTP %s (tentative %d/3)", lettre, page, e.response.status_code, attempt)
            except httpx.RequestError as e:
                log.warning("[%s p%d] Erreur réseau : %s (tentative %d/3)", lettre, page, e, attempt)
            await asyncio.sleep(2 ** attempt)
    log.error("[%s p%d] Echec definitif.", lettre, page)
    return None


# ---------------------------------------------------------------------------
# Scraping par lettre
# ---------------------------------------------------------------------------

async def scraper_lettre(
    lettre: str,
    client: httpx.AsyncClient,
    sem_lettres: asyncio.Semaphore,
) -> None:
    """
    Besoin : scraper toutes les pages d'une lettre et sauvegarder en CSV.
    Choix  :
        - Page 1 scrappée en premier pour obtenir le nb total -> calcul nb pages
        - Pages 2..N lancées en parallèle via asyncio.gather avec semaphore dédié
        - Sauvegarde CSV immédiate par lettre : résilience aux interruptions
    Impact : chaque lettre = 1 fichier CSV dans /data -> pas de perte si crash
    """
    async with sem_lettres:
        log.info("[%s] Debut", lettre)

        sem_pages = asyncio.Semaphore(MAX_CONCURRENT_PAGES)

        # -- Page 1 : récupération du nombre total de prénoms
        soup_p1 = await fetch_page(client, lettre, 1, sem_pages)
        if soup_p1 is None:
            log.error("[%s] Page 1 inaccessible, lettre ignoree.", lettre)
            return

        nb_total   = parser_nombre_prenoms(soup_p1)
        nb_pages   = max(1, math.ceil(nb_total / PRENOMS_PAR_PAGE))
        prenoms_p1 = parser_prenoms_page(soup_p1)

        # Affichage du premier prénom scrappé en live (validation rapide)
        if prenoms_p1:
            log.info(
                "[%s] Page 1 | Premier prenom : '%s' | Total attendu : %d prenoms / %d pages",
                lettre, prenoms_p1[0]["prenom"], nb_total, nb_pages,
            )

        tous = list(prenoms_p1)

        # -- Pages 2..N en parallèle
        if nb_pages > 1:
            taches = [
                fetch_page(client, lettre, page, sem_pages)
                for page in range(2, nb_pages + 1)
            ]
            soups = await asyncio.gather(*taches)

            for idx, soup in enumerate(soups, start=2):
                if soup is None:
                    continue
                prenoms = parser_prenoms_page(soup)
                if prenoms:
                    log.info("[%s] Page %d | Premier prenom : '%s'", lettre, idx, prenoms[0]["prenom"])
                tous.extend(prenoms)

        # -- Déduplication et export CSV
        df = pd.DataFrame(tous, columns=["prenom", "url", "sexe"])
        nb_avant = len(df)
        df = df.drop_duplicates(subset=["url"])
        if nb_avant != len(df):
            log.info("[%s] Deduplication : %d -> %d lignes", lettre, nb_avant, len(df))

        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(csv_path(lettre), index=False, encoding="utf-8-sig")
        log.info("[%s] Termine : %d prenoms -> %s", lettre, len(df), csv_path(lettre))


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------

async def run(lettres: list[str]) -> None:
    """
    Besoin : orchestrer le scraping de plusieurs lettres avec concurrence contrôlée.
    Choix  : un seul AsyncClient partagé (pool TCP), semaphore global sur les lettres.
    Usage  : await run(['A', 'B', 'C'])
    """
    sem_lettres = asyncio.Semaphore(MAX_CONCURRENT_LETTRES)

    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
        taches = [
            scraper_lettre(lettre, client, sem_lettres)
            for lettre in lettres
        ]
        await asyncio.gather(*taches)


def choisir_lettre_depart() -> str:
    """
    Besoin : permettre à l'utilisateur de reprendre le scraping à une lettre donnée.
    Choix  : input() simple avec validation, sans dépendance externe.
    Usage  : interactif en CLI
    """
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    while True:
        rep = input("Lettre de depart (A-Z, Entree = A) : ").strip().upper()
        if rep == "":
            return "A"
        if rep in alphabet:
            return rep
        print(f"  Lettre invalide : '{rep}'. Entrer une lettre entre A et Z.")


def main() -> None:
    alphabet      = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    depart        = choisir_lettre_depart()
    idx_debut     = alphabet.index(depart)
    lettres       = alphabet[idx_debut:]

    # Filtrer les lettres dont le CSV existe déjà (reprise automatique)
    lettres_a_traiter = [l for l in lettres if not lettre_deja_scrapee(l)]
    lettres_ignorees  = [l for l in lettres if lettre_deja_scrapee(l)]

    if lettres_ignorees:
        log.info("Lettres deja scrapees (ignorees) : %s", ", ".join(lettres_ignorees))

    if not lettres_a_traiter:
        log.info("Toutes les lettres demandees sont deja scrapees. Rien a faire.")
        return

    log.info(
        "Lettres a traiter : %s | Concurrence : %d lettres x %d pages",
        ", ".join(lettres_a_traiter),
        MAX_CONCURRENT_LETTRES,
        MAX_CONCURRENT_PAGES,
    )

    asyncio.run(run(lettres_a_traiter))

    # Rapport final
    csv_produits = [csv_path(l) for l in lettres_a_traiter if lettre_deja_scrapee(l)]
    total = sum(len(pd.read_csv(f)) for f in csv_produits if os.path.isfile(f))
    log.info("Session terminee | %d prénoms collectes | CSV dans /%s/", total, DATA_DIR)


if __name__ == "__main__":
    main()