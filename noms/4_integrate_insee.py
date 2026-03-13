import json
import os
from collections import defaultdict
from dbfread import DBF

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INSEE_FILE = os.path.join(DATA_DIR, "noms_insee", "Nat2008.dbf")
NOMS_FINAL = os.path.join(DATA_DIR, "3_noms_final.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "4_noms_final_insee.json")

def main():
    print(f"Loading INSEE data from {INSEE_FILE}...")
    table = DBF(INSEE_FILE, load=True, encoding='latin-1') # dbf are often latin-1 or cp1252
    
    insee_data = {}
    periods = [
        '_1891_1900', '_1901_1910', '_1911_1920', '_1921_1930', 
        '_1931_1940', '_1941_1950', '_1951_1960', '_1961_1970', 
        '_1971_1980', '_1981_1990', '_1991_2000'
    ]
    
    for record in table.records:
        nom = str(record.get('NOM', '')).strip().lower()
        if not nom:
            continue
            
        historique = {}
        total = 0
        for p in periods:
            val = record.get(p, 0)
            if val is None: val = 0
            historique[p.strip('_')] = int(val)
            total += int(val)
            
        insee_data[nom] = {
            "nombre_total": total,
            "historique": historique
        }
        
    print(f"Loaded {len(insee_data)} names from INSEE.")
    
    print(f"Loading {NOMS_FINAL}...")
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
            
    print(f"Matched {matched} out of {len(noms_final)} names.")
    
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(noms_final, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
