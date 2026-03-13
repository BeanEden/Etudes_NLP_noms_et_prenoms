"""
4_integrate_insee.py — Enrichissement des noms avec les données INSEE
======================================================================
Entrée  : ./data/3_noms_final.json
          ./data/noms_insee/Nat2008.dbf  (fichier INSEE au format DBF)
Sortie  : ./data/4_noms_final_insee.json

Pour chaque nom présent dans la base, enrichit la fiche avec un historique
de naissances par période décennale (1891–2000) ainsi que le total cumulé,
en jointure sur le nom normalisé (minuscules, strip).

Dépendances :
    pip install dbfread
"""

import json
import logging
import os

from dbfread import DBF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INSEE_FILE = os.path.join(DATA_DIR, "noms_insee", "Nat2008.dbf")
NOMS_FINAL = os.path.join(DATA_DIR, "3_noms_final.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "4_noms_final_insee.json")

def main() -> None:
    """
    Charge le DBF INSEE (Nat2008.dbf), construit un index nom -> historique décennal,
    puis enrichit chaque entrée de noms_final.json avec le champ 'insee_data'.
    Les noms sans correspondance reçoivent insee_data=None.
    """
    log.info("Chargement des données INSEE depuis %s...", INSEE_FILE)
    # Les fichiers DBF INSEE sont encodés en latin-1 (cp1252 accepté aussi)
    table = DBF(INSEE_FILE, load=True, encoding="latin-1")

    periods = [
        "_1891_1900", "_1901_1910", "_1911_1920", "_1921_1930",
        "_1931_1940", "_1941_1950", "_1951_1960", "_1961_1970",
        "_1971_1980", "_1981_1990", "_1991_2000",
    ]

    insee_data = {}
    for record in table.records:
        nom = str(record.get("NOM", "")).strip().lower()
        if not nom:
            continue

        historique = {}
        total = 0
        for p in periods:
            val = record.get(p, 0) or 0
            historique[p.strip("_")] = int(val)
            total += int(val)

        insee_data[nom] = {
            "nombre_total": total,
            "historique":   historique,
        }

    log.info("%d noms chargés depuis l'INSEE.", len(insee_data))

    log.info("Chargement de %s...", NOMS_FINAL)
    with open(NOMS_FINAL, "r", encoding="utf-8") as f:
        noms_final = json.load(f)

    matched = 0
    for entry in noms_final:
        nom = entry.get("nom", "").strip().lower()
        if nom in insee_data:
            entry["insee_data"] = insee_data[nom]
            matched += 1
        else:
            entry["insee_data"] = None

    log.info("Jointure : %d / %d noms appariés.", matched, len(noms_final))

    log.info("Export -> %s", OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(noms_final, f, ensure_ascii=False, indent=2)

    log.info("Terminé.")

if __name__ == "__main__":
    main()
