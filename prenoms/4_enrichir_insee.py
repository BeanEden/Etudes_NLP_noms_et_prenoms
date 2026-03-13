"""
enrichir_insee.py — Jointure prénoms_final.json × INSEE parquet
================================================================
Entrées  : ./data/prenoms_final.json
           ./data/prenoms_insee/prenoms-2024.parquet
Sorties  : ./data/prenoms_stats.parquet   (stats par prénom × année × dept)
           ./data/prenoms_tendances.json  (courbes agrégées par prénom, prêt JS)

Structure prenoms_stats.parquet (colonnes) :
    prenom, annee, dept, count, id_groupe_total

Structure prenoms_tendances.json :
    {
        "jean": {
            "national": [{"annee": 1900, "count": 12300}, ...],
            "par_dept": {"75": [{"annee": 1980, "count": 450}, ...], ...},
            "top_depts": ["75", "69", "13"],   # 3 depts avec le plus de naissances
            "pic_annee": 1954,
            "total":     890432,
        },
        ...
    }

Normalisation des prénoms INSEE -> même normaliser() que prepare_prenoms.py
pour garantir la jointure (accents, casse, tirets).

Filtre : on exclut '_PRENOMS_RARES' et les counts < MIN_COUNT.

Dépendances :
    pip install pandas pyarrow tqdm
"""

import json
import logging
import os
import re
import unicodedata

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PRENOMS  = "prenoms/data/4_prenoms_final.json"
INPUT_PARQUET  = "prenoms/data/prenoms_insee/prenoms-2024.parquet"
OUTPUT_PARQUET = "prenoms/data/5_prenoms_stats.parquet"
OUTPUT_JSON    = "prenoms/data/5_prenoms_tendances.json"

MIN_COUNT      = 3    # seuil minimal de naissances (anonymisation INSEE)
TOP_DEPTS      = 5    # nombre de depts à exposer dans les tendances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation (copie de prepare_prenoms.py — sans dépendance croisée)
# ---------------------------------------------------------------------------

_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_NON_ALPHA   = re.compile(r"[^a-z\s\-]")


def normaliser(prenom: str) -> str:
    p = unicodedata.normalize("NFD", prenom)
    p = "".join(c for c in p if unicodedata.category(c) != "Mn")
    p = p.lower().strip()
    p = _RE_NON_ALPHA.sub(" ", p)
    return _RE_MULTI_SPACE.sub(" ", p).strip()


# ---------------------------------------------------------------------------
# Chargement et inspection du parquet INSEE
# ---------------------------------------------------------------------------

def charger_insee(path: str) -> pd.DataFrame:
    """
    Charge le parquet INSEE au schéma réel :
        sexe, prenom, periode, niveau_geographique, geographie, valeur

    Logique de filtrage :
        - On garde niveau_geographique == 'DEP' pour avoir la maille département.
          Les autres valeurs ('REG', 'FRANCE', etc.) seraient des doublons agrégés.
        - periode  -> annee  (int)
        - geographie -> dept (str, zero-padded)
        - valeur   -> count  (int)
        - On exclut les prénoms rares (valeur souvent masquée à NaN ou 0)
        - On filtre count < MIN_COUNT
    """
    df = pd.read_parquet(path)
    log.info("Parquet INSEE chargé : %s colonnes, %d lignes", list(df.columns), len(df))

    # Inspection des valeurs de niveau_geographique pour choisir le bon filtre
    if "niveau_geographique" in df.columns:
        niveaux = df["niveau_geographique"].value_counts().to_dict()
        log.info("niveau_geographique : %s", niveaux)
        # On garde uniquement le niveau département
        niveaux_dep = [n for n in niveaux if str(n).upper() in ("DEP", "DEPARTEMENT", "D")]
        if niveaux_dep:
            df = df[df["niveau_geographique"].isin(niveaux_dep)]
            log.info("Filtrage niveau DEP : %d lignes conservées", len(df))
        else:
            log.warning("Niveau 'DEP' introuvable — toutes les lignes conservées. "
                        "Valeurs disponibles : %s", list(niveaux.keys()))

    # Renommage vers schéma interne
    df = df.rename(columns={
        "prenom":      "prenom_insee",
        "periode":     "annee",
        "geographie":  "dept",
        "valeur":      "count",
    })

    # Nettoyage periode : peut contenir des plages "1900-1910" -> on prend l'année de début
    df["annee"] = (
        df["annee"].astype(str)
        .str.extract(r"(\d{4})", expand=False)
    )
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    df = df.dropna(subset=["annee", "count"])
    df = df[df["count"] >= MIN_COUNT]
    df["annee"] = df["annee"].astype(int)
    df["count"] = df["count"].astype(int)
    df["dept"]  = df["dept"].astype(str).str.strip().str.zfill(2)

    # Exclusion prénoms rares (marqueur INSEE)
    df = df[~df["prenom_insee"].astype(str).str.startswith("_")]

    log.info("Après filtrage : %d lignes", len(df))
    return df


# ---------------------------------------------------------------------------
# Jointure avec prenoms_final
# ---------------------------------------------------------------------------

def joindre(df_insee: pd.DataFrame, prenoms_final: list) -> pd.DataFrame:
    """
    Normalise les prénoms INSEE et joint avec l'index prenoms_final.
    Ajoute id_groupe_total pour permettre les agrégations par famille.
    """
    # Index normalisation -> id_groupe_total
    index_groupe = {d["prenom"]: d["id_groupe_total"] for d in prenoms_final}

    df_insee["prenom_norm"] = df_insee["prenom_insee"].apply(normaliser)
    df_insee["id_groupe_total"] = df_insee["prenom_norm"].map(index_groupe)

    n_total    = len(df_insee)
    n_joint    = df_insee["id_groupe_total"].notna().sum()
    n_non_joint = n_total - n_joint

    log.info(
        "Jointure : %d / %d lignes jointes (%.1f%% couverts | %d hors base)",
        n_joint, n_total, 100 * n_joint / n_total if n_total else 0, n_non_joint,
    )

    df_joint = df_insee[df_insee["id_groupe_total"].notna()].copy()
    df_joint["id_groupe_total"] = df_joint["id_groupe_total"].astype(int)
    df_joint = df_joint.rename(columns={"prenom_norm": "prenom"})
    return df_joint[["prenom", "annee", "dept", "count", "id_groupe_total"]]


# ---------------------------------------------------------------------------
# Construction des tendances JSON (pour les graphiques Flask/JS)
# ---------------------------------------------------------------------------

def construire_tendances(df: pd.DataFrame) -> dict:
    """
    Pour chaque prénom normalisé :
        - Série nationale annuelle (groupby prenom + annee)
        - Série par département (groupby prenom + dept + annee)
        - Top N départements (sum total)
        - Année de pic national
        - Total national cumulé
    """
    tendances = {}

    prenoms_uniques = df["prenom"].unique()
    log.info("Construction tendances pour %d prénoms...", len(prenoms_uniques))

    for prenom in tqdm(prenoms_uniques, desc="Tendances", unit="prénom"):
        sub = df[df["prenom"] == prenom]

        # National
        national = (
            sub.groupby("annee")["count"].sum()
            .reset_index()
            .sort_values("annee")
        )
        national_list = [
            {"annee": int(r.annee), "count": int(r.count)}
            for r in national.itertuples()
        ]

        # Total + pic
        total    = int(national["count"].sum())
        pic_idx  = national["count"].idxmax() if not national.empty else None
        pic_annee = int(national.loc[pic_idx, "annee"]) if pic_idx is not None else None

        # Par département
        par_dept_raw = (
            sub.groupby(["dept", "annee"])["count"].sum()
            .reset_index()
            .sort_values(["dept", "annee"])
        )

        # Top depts par volume total
        top_depts_df = (
            sub.groupby("dept")["count"].sum()
            .nlargest(TOP_DEPTS)
            .index.tolist()
        )

        par_dept = {}
        for dept in top_depts_df:
            dept_data = par_dept_raw[par_dept_raw["dept"] == dept]
            par_dept[dept] = [
                {"annee": int(r.annee), "count": int(r.count)}
                for r in dept_data.itertuples()
            ]

        tendances[prenom] = {
            "national":  national_list,
            "par_dept":  par_dept,
            "top_depts": top_depts_df,
            "pic_annee": pic_annee,
            "total":     total,
        }

    return tendances


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    for path in [INPUT_PRENOMS, INPUT_PARQUET]:
        if not os.path.isfile(path):
            log.error("Fichier introuvable : %s", path)
            return

    os.makedirs("data", exist_ok=True)

    with open(INPUT_PRENOMS, "r", encoding="utf-8") as f:
        prenoms_final = json.load(f)
    log.info("%d prénoms chargés depuis %s", len(prenoms_final), INPUT_PRENOMS)

    df_insee = charger_insee(INPUT_PARQUET)
    df_joint = joindre(df_insee, prenoms_final)

    # Export parquet enrichi
    df_joint.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("Export : %s (%d lignes)", OUTPUT_PARQUET, len(df_joint))

    # Export tendances JSON
    tendances = construire_tendances(df_joint)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tendances, f, ensure_ascii=False, separators=(",", ":"))
    taille_mo = os.path.getsize(OUTPUT_JSON) / 1024 / 1024
    log.info("Export : %s (%.1f Mo, %d prénoms)", OUTPUT_JSON, taille_mo, len(tendances))


if __name__ == "__main__":
    main()