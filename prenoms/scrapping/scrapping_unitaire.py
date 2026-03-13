"""
Scraper Phase 2 - prenoms.com
Objectif  : pour chaque URL du fichier prenoms.csv, scraper signification / caractere / histoire
Entrée    : /data/prenoms.csv (colonnes : prenom, url, sexe)
Sortie    : /data/prenoms_detail.json  (array JSON, un objet par prénom)

Dépendances :
    pip install httpx beautifulsoup4 pandas

Choix techniques :
    - httpx async : cohérence avec la phase 1, perf I/O sur N milliers d'URLs
    - Semaphore unique : pas de parallélisme multi-source ici, juste contrôle du débit
    - JSON array plat : format optimal pour NLP aval (un doc = un objet, textes bruts)
    - Reprise : les URLs déjà présentes dans le JSON de sortie sont sautées au redémarrage
    - Flush incrémental : écriture après chaque batch pour limiter les pertes en cas de crash
"""

import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV   = "prenoms/data/prenoms.csv"
OUTPUT_JSON = "prenoms/data/1_prenoms_detail.json"

# Concurrence : ~10 requêtes simultanées, raisonnable pour un site grand public
MAX_CONCURRENT = 10

DELAY_MIN = 0.5
DELAY_MAX = 1.5

# Nombre d'URLs traitées entre chaque flush sur disque
BATCH_FLUSH = 100

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
# Parsing HTML
# ---------------------------------------------------------------------------

def extraire_section(soup: BeautifulSoup, section_id: str) -> str:
    """
    Besoin : extraire le texte d'une section identifiée par l'id de son h2.
    Choix  : on cherche le h2 par id, puis on collecte tous les <p> siblings
             jusqu'au prochain h2 (ou fin de parent). Couvre les cas multi-paragraphes.
    Usage  : extraire_section(soup, "signification") -> "Le prénom Mamadou..."

    Retourne une chaîne vide si la section est absente (données manquantes tolérées).
    """
    h2 = soup.find("h2", id=section_id)
    if h2 is None:
        return ""

    paragraphes = []
    for sibling in h2.find_next_siblings():
        # Arrêt dès le prochain titre de section
        if sibling.name in ("h2", "h3"):
            break
        if sibling.name == "p":
            texte = sibling.get_text(separator=" ", strip=True)
            if texte:
                paragraphes.append(texte)

    return " ".join(paragraphes)


def parser_page_prenom(html: str) -> dict:
    """
    Besoin : parser une page de prénom et retourner les trois champs texte.
    Choix  : BeautifulSoup html.parser (pas de lxml requis, dépendance minimale).
    Usage  : parser_page_prenom(response.text) -> {signification, caractere, histoire}
    """
    soup = BeautifulSoup(html, "html.parser")
    return {
        "signification": extraire_section(soup, "signification"),
        "caractere":     extraire_section(soup, "caractere"),
        "histoire":      extraire_section(soup, "histoire"),
        "etymologie":    extraire_section(soup, "etymologie"),
        "provenance":    extraire_section(soup, "provenance"),
    }


# ---------------------------------------------------------------------------
# Requêtes async
# ---------------------------------------------------------------------------

async def fetch_prenom(

    client: httpx.AsyncClient,
    row: dict,
    sem: asyncio.Semaphore,
    compteur: list,
    total: int,
) -> Optional[dict]:
    """
    Besoin : scraper une page de prénom et retourner l'objet complet.
    Choix  : retry x3 avec backoff exponentiel, même pattern que phase 1.
             compteur mutable [int] pour log de progression sans état global.
    Usage  : await fetch_prenom(client, {"prenom": "mamadou", "url": "...", "sexe": "garcon"}, sem, compteur, total)
    """
    async with sem:
        await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        for attempt in range(1, 4):
            try:
                r = await client.get(row["url"], timeout=15)
                r.raise_for_status()

                detail = parser_page_prenom(r.text)
                compteur[0] += 1

                # Log de progression tous les 50 prénoms
                if compteur[0] % 50 == 0 or compteur[0] == total:
                    log.info("Progression : %d / %d (%.1f%%)", compteur[0], total, 100 * compteur[0] / total)

                return {
                    "prenom":        row["prenom"],
                    "url":           row["url"],
                    "sexe":          row["sexe"],
                    "signification": detail["signification"],
                    "caractere":     detail["caractere"],
                    "histoire":      detail["histoire"],
                    "etymologie":   detail["etymologie"],
                    "provenance":   detail["provenance"],
                }

            except httpx.HTTPStatusError as e:
                log.warning("[%s] HTTP %s (tentative %d/3)", row["prenom"], e.response.status_code, attempt)
            except httpx.RequestError as e:
                log.warning("[%s] Erreur réseau : %s (tentative %d/3)", row["prenom"], e, attempt)

            await asyncio.sleep(2 ** attempt)

    log.error("[%s] Echec définitif, URL ignorée : %s", row["prenom"], row["url"])
    return None


# ---------------------------------------------------------------------------
# Gestion de la reprise
# ---------------------------------------------------------------------------

def charger_deja_traites(output_path: str) -> set:
    """
    Besoin : identifier les URLs déjà scrapées pour éviter de les retraiter.
    Choix  : lecture du JSON existant, index sur 'url' (clé naturelle stable).
    Usage  : appelé au démarrage, retourne un set d'URLs déjà présentes.
    """
    if not os.path.isfile(output_path):
        return set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        urls = {item["url"] for item in data if "url" in item}
        log.info("Reprise : %d prénoms déjà traités chargés depuis %s", len(urls), output_path)
        return urls
    except (json.JSONDecodeError, KeyError):
        log.warning("JSON de sortie corrompu ou vide, reprise depuis zéro.")
        return set()


def flush_json(output_path: str, resultats: list) -> None:
    """
    Besoin : persister les résultats sur disque de façon incrémentale.
    Choix  : réécriture complète du fichier à chaque flush (simple, pas de corruption
             partielle possible). Acceptable tant que le fichier reste < 50 Mo.
    Impact maintenabilité : si le volume dépasse 50 Mo, passer à un append ligne par ligne
                            en JSON-Lines (.jsonl) pour éviter la réécriture complète.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run(rows: list[dict], deja_traites: set) -> list[dict]:
    """
    Besoin : scraper toutes les URLs en async avec flush incrémental par batch.
    Choix  : asyncio.gather par batch de BATCH_FLUSH pour limiter la mémoire
             et permettre les sauvegardes intermédiaires.
    """
    # Chargement du JSON existant pour y ajouter les nouveaux résultats
    resultats_existants = []
    if os.path.isfile(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            try:
                resultats_existants = json.load(f)
            except json.JSONDecodeError:
                resultats_existants = []

    # Filtrage des URLs non encore traitées
    a_traiter = [r for r in rows if r["url"] not in deja_traites]
    total     = len(a_traiter)

    if total == 0:
        log.info("Aucune nouvelle URL à traiter.")
        return resultats_existants

    log.info("%d URLs à scraper (%d déjà traitées ignorées)", total, len(deja_traites))

    sem       = asyncio.Semaphore(MAX_CONCURRENT)
    compteur  = [0]  # liste mutable pour passage par référence dans les coroutines
    tous      = list(resultats_existants)

    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
        # Traitement par batch pour flush intermédiaire
        for i in range(0, total, BATCH_FLUSH):
            batch = a_traiter[i : i + BATCH_FLUSH]
            taches = [
                fetch_prenom(client, row, sem, compteur, total)
                for row in batch
            ]
            resultats_batch = await asyncio.gather(*taches)

            nouveaux = [r for r in resultats_batch if r is not None]
            tous.extend(nouveaux)

            log.info(
                "Flush batch %d-%d : %d résultats ajoutés | Total JSON : %d",
                i + 1, min(i + BATCH_FLUSH, total), len(nouveaux), len(tous),
            )
            flush_json(OUTPUT_JSON, tous)

    return tous


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    # -- Chargement du CSV source
    if not os.path.isfile(INPUT_CSV):
        log.error("Fichier source introuvable : %s", INPUT_CSV)
        return

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    colonnes_requises = {"prenom", "url", "sexe"}
    if not colonnes_requises.issubset(df.columns):
        log.error("Colonnes manquantes dans %s. Attendu : %s", INPUT_CSV, colonnes_requises)
        return

    # Nettoyage défensif : on retire les lignes sans URL
    df = df.dropna(subset=["url"])
    rows = df[["prenom", "url", "sexe"]].to_dict(orient="records")
    log.info("%d prénoms chargés depuis %s", len(rows), INPUT_CSV)

    # -- Reprise : URLs déjà dans le JSON de sortie
    deja_traites = charger_deja_traites(OUTPUT_JSON)

    # -- Scraping async
    resultats = asyncio.run(run(rows, deja_traites))

    # -- Rapport final
    df_out = pd.DataFrame(resultats)
    vides  = (df_out[["signification", "caractere", "histoire","etymologie","provenance"]] == "").all(axis=1).sum()
    log.info(
        "Termine | %d prénoms dans le JSON | %d sans aucune section (pages vides ou structure différente)",
        len(resultats), vides,
    )
    log.info("Sortie : %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()