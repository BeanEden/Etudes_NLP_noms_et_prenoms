"""
Flask — Interface noms de famille
==================================
Charge noms_final.json et groupes_finals.json en mémoire au démarrage.
Routes :
    /                   -> recherche
    /nom/<nom>          -> fiche détaillée
    /api/search?q=...   -> autocomplétion JSON
    /api/nom/<nom>      -> données JSON pour la carte Leaflet
"""

import json
import os
import re
import sys
import subprocess
import unicodedata
import numpy as np
from collections import defaultdict
import nbformat
from nbconvert import HTMLExporter
from flask import Flask, render_template, request, jsonify, abort, g

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Chargement en mémoire au démarrage
# ---------------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOMS_DIR = os.path.join(BASE_DIR, "noms", "data")
PRENOMS_DIR = os.path.join(BASE_DIR, "prenoms", "data")

def _charger(filename, directory, default=None):
    if default is None: default = []
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return default
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return default

print("Loading noms data...")
_noms_final    = _charger("4_noms_final_insee.json", NOMS_DIR)
_groupes_final = _charger("3_groupes_finals.json", NOMS_DIR)

# Index principal : nom normalisé -> dict complet
INDEX_NOMS: dict = {d["nom"]: d for d in _noms_final}
INDEX_GROUPES: dict = {g["id_groupe_total"]: g for g in _groupes_final}
NOMS_TRIES = sorted(INDEX_NOMS.keys())

print("Loading prenoms data...")
_prenoms_final     = _charger("4_prenoms_final.json", PRENOMS_DIR)
_groupes_prenoms   = _charger("4_groupes_finals_prenoms.json", PRENOMS_DIR)
_prenoms_tendances = _charger("5_prenoms_tendances.json", PRENOMS_DIR, default={})

INDEX_PRENOMS: dict = {d["prenom"]: d for d in _prenoms_final}
PRENOMS_TRIES = sorted(INDEX_PRENOMS.keys())

# ---------------------------------------------------------------------------
# Coordonnées géographiques par label géo (centroïdes approximatifs)
# ---------------------------------------------------------------------------

GEO_COORDS = {
    "bretagne":        (48.20, -2.93),
    "normandie":       (49.18,  0.37),
    "alsace":          (48.47,  7.56),
    "lorraine":        (48.87,  6.18),
    "provence":        (43.93,  5.72),
    "languedoc":       (43.61,  3.88),
    "gascogne":        (43.80,  0.57),
    "bourgogne":       (47.05,  4.85),
    "auvergne":        (45.45,  3.17),
    "savoie":          (45.57,  6.35),
    "dauphiné":        (45.19,  5.72),
    "poitou":          (46.58,  0.34),
    "anjou":           (47.47, -0.55),
    "touraine":        (47.39,  0.69),
    "champagne":       (49.04,  4.02),
    "picardie":        (49.89,  2.30),
    "flandre":         (50.85,  2.71),
    "pays basque":     (43.29, -1.65),
    "catalogne":       (41.83,  2.13),
    "jura":            (46.67,  5.55),
    "franche-comté":   (47.28,  6.02),
    "île-de-france":   (48.85,  2.35),
    "france":          (46.60,  1.88),
    "allemagne":       (51.17, 10.45),
    "angleterre":      (52.36, -1.17),
    "italie":          (42.83, 12.83),
    "espagne":         (40.42, -3.70),
    "portugal":        (39.40, -8.22),
    "belgique":        (50.85,  4.35),
    "suisse":          (46.82,  8.23),
    "pays-bas":        (52.13,  5.29),
    "scandinavie":     (62.00, 15.00),
    "irlande":         (53.41, -8.24),
    "pologne":         (52.07, 19.48),
    "maghreb":         (30.00,  3.00),
    "maroc":           (31.79, -7.09),
    "algérie":         (28.03,  1.66),
    "tunisie":         (33.89,  9.54),
    "nord-pas-de-calais": (50.48, 2.79),
}

# ---------------------------------------------------------------------------
# Extraction de date/siècle depuis le texte brut
# ---------------------------------------------------------------------------

_RE_SIECLE  = re.compile(r"\b(X{0,3}(?:IX|IV|VI{0,3}|I{1,3})e?)\s*si[èe]cle\b", re.IGNORECASE)
_RE_ANNEE   = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_SIECLE_MAP = {
    "ier": 1, "ie": 1, "i": 1,
    "iie": 2, "ii": 2,
    "iiie": 3, "iii": 3,
    "ive": 4, "iv": 4,
    "ve": 5, "v": 5,
    "vie": 6, "vi": 6,
    "viie": 7, "vii": 7,
    "viiie": 8, "viii": 8,
    "ixe": 9, "ix": 9,
    "xe": 10, "x": 10,
    "xie": 11, "xi": 11,
    "xiie": 12, "xii": 12,
    "xiiie": 13, "xiii": 13,
    "xive": 14, "xiv": 14,
    "xve": 15, "xv": 15,
    "xvie": 16, "xvi": 16,
    "xviie": 17, "xvii": 17,
    "xviiie": 18, "xviii": 18,
    "xixe": 19, "xix": 19,
    "xxe": 20, "xx": 20,
}

def extraire_date(texte: str) -> dict:
    """
    Extrait la première référence temporelle du texte brut.
    Retourne {"type": "siecle"|"annee"|None, "valeur": int|None, "label": str}
    """
    if not texte:
        return {"type": None, "valeur": None, "label": ""}

    m = _RE_SIECLE.search(texte)
    if m:
        code  = m.group(1).lower().replace(" ", "")
        siecle = _SIECLE_MAP.get(code)
        if siecle:
            annee_approx = (siecle - 1) * 100 + 50
            return {"type": "siecle", "valeur": annee_approx, "label": f"{m.group(1).upper()} siècle"}

    m = _RE_ANNEE.search(texte)
    if m:
        annee = int(m.group(1))
        return {"type": "annee", "valeur": annee, "label": str(annee)}

    return {"type": None, "valeur": None, "label": ""}

# ---------------------------------------------------------------------------
# Préparation des données carte
# ---------------------------------------------------------------------------

def preparer_carte(fiche: dict) -> dict:
    """
    Prépare les données GeoJSON-like pour Leaflet :
        - Point principal : coord du nom affiché
        - Points connexes : coords des noms du groupe (si geo disponible)
        - Lignes : connexions entre le nom principal et chaque nom lié

    Si aucune géo n'est disponible, retourne un dict vide (carte masquée).
    """
    geo_label = fiche.get("geo", "")
    coords_principal = GEO_COORDS.get(geo_label)

    noms_groupe = fiche.get("noms_groupe", [])
    connexions = []

    for nom_lie in noms_groupe:
        if nom_lie == fiche["nom"]:
            continue
        entry = INDEX_NOMS.get(nom_lie, {})
        geo_lie = entry.get("geo", "")
        coords_lie = GEO_COORDS.get(geo_lie) if geo_lie else None

        # Si le nom lié n'a pas de géo propre, on utilise celle du nom principal
        coords_effective = coords_lie or coords_principal
        if coords_effective:
            connexions.append({
                "nom":         entry.get("nom_original", nom_lie),
                "lat":         coords_effective[0],
                "lng":         coords_effective[1],
                "same_point":  coords_effective == coords_principal,
            })

    return {
        "principal": {
            "lat":   coords_principal[0] if coords_principal else None,
            "lng":   coords_principal[1] if coords_principal else None,
            "label": fiche.get("nom_original", fiche["nom"]),
        },
        "connexions": connexions,
        "has_geo":    coords_principal is not None,
    }

# ---------------------------------------------------------------------------
# Préparation des données graphe (Nœuds et Liens)
# ---------------------------------------------------------------------------

def preparer_graphe(fiche: dict) -> dict:
    """
    Prépare les données pour le graphe réseau (nodes, edges) du groupe de noms.
    Les tailles des nœuds dépendent de insee_data['nombre_total'] si disponible.
    """
    nodes = []
    edges = []
    
    noms_groupe = fiche.get("noms_groupe", [])
    if not noms_groupe:
        # Aucun lié, on s'ajoute quand même tout seul
        noms_groupe = [fiche["nom"]]
        
    for idx, nom_g in enumerate(noms_groupe):
        entry = INDEX_NOMS.get(nom_g, {})
        
        # Taille par défaut ou calculée depuis insee
        size = 10
        insee = entry.get("insee_data", {})
        if insee and isinstance(insee, dict):
            nb_total = insee.get("nombre_total", 0)
            if nb_total > 0:
                # Echelle log ou racine pour pas exploser les tailles
                size = max(10, int(nb_total ** 0.5)) + 5
                
        is_principal = (nom_g == fiche["nom"])
        
        nodes.append({
            "id": nom_g,
            "label": entry.get("nom_original", nom_g),
            "value": size, # utilisé par vis.js pour ajuster la taille
            "group": "principal" if is_principal else "lie",
            "title": f"Fréquence: {nb_total if insee else 'Inconnue'}"
        })
        
        # On relie tout au nom principal (étoile) ou on relie les noms consécutifs (cercle)
        # Ici on fait une étoile depuis le nom principal demandé
        if not is_principal and idx != 0:
            edges.append({
                "from": fiche["nom"],
                "to": nom_g
            })
            
    return {
        "nodes": nodes,
        "edges": edges
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    q       = request.args.get("q", "").strip().lower()
    error   = request.args.get("error", "")
    f_type  = request.args.get("filter", "all").strip().lower()
    results = []
    
    if q and len(q) >= 2:
        # Détection de recherche combinée (Nom + Prénom ou Prénom + Nom)
        parts = q.split()
        if len(parts) == 2 and f_type == "all":
            p1, p2 = parts[0], parts[1]
            # Cas 1: p1 = Prénom, p2 = Nom
            if p1 in INDEX_PRENOMS and p2 in INDEX_NOMS:
                results.append({
                    "type_entite": "compare",
                    "id1": p1, "type1": "prenom", "nom_original_1": INDEX_PRENOMS[p1]["prenom_original"],
                    "id2": p2, "type2": "nom", "nom_original_2": INDEX_NOMS[p2].get("nom_original", p2),
                    "texte_resume": "Comparer ces deux entités conjointement sur une chronologie unifiée."
                })
            # Cas 2: p1 = Nom, p2 = Prénom
            elif p1 in INDEX_NOMS and p2 in INDEX_PRENOMS:
                results.append({
                    "type_entite": "compare",
                    "id1": p1, "type1": "nom", "nom_original_1": INDEX_NOMS[p1].get("nom_original", p1),
                    "id2": p2, "type2": "prenom", "nom_original_2": INDEX_PRENOMS[p2]["prenom_original"],
                    "texte_resume": "Comparer ces deux entités conjointement sur une chronologie unifiée."
                })

        # Recherche Noms
        if f_type in ["all", "nom"]:
            for n in [n for n in NOMS_TRIES if q in n][:20]:
                r = dict(INDEX_NOMS[n])
                r["type_entite"] = "nom"
                results.append(r)
            
        # Recherche Prénoms
        if f_type in ["all", "prenom"]:
            for p in [p for p in PRENOMS_TRIES if q in p][:20]:
                r = dict(INDEX_PRENOMS[p])
                r["type_entite"] = "prenom"
                results.append(r)
            
        # Tri combiné: Les résultats 'compare' d'abord, puis correspondance stricte, puis le reste
        results = sorted(results, key=lambda x: (
            0 if x.get("type_entite") == "compare" else 1,
            x.get("nom", x.get("prenom", "")) != q,
            x.get("nom", x.get("prenom", ""))
        ))
            
    return render_template("search.html", query=q, results=results, error=error, filter=f_type)


@app.route("/nom/<nom>")
def fiche(nom):
    nom = nom.strip().lower()
    data = INDEX_NOMS.get(nom)
    if not data:
        return render_template("search.html", query=nom, type="nom", error=f"Aucun nom similaire à « {nom} » n'a été trouvé dans la base de données.")
    carte   = preparer_carte(data)
    chrono  = extraire_date(data.get("origine_brute", ""))
    graphe  = preparer_graphe(data)
    return render_template("fiche.html", fiche=data, carte=carte, chrono=chrono, graphe=graphe)


@app.route("/prenom/<prenom>")
def fiche_prenom(prenom):
    prenom = prenom.strip().lower()
    data = INDEX_PRENOMS.get(prenom)
    if not data:
        return render_template("search.html", query=prenom, type="prenom", error=f"Aucun prénom similaire à « {prenom} » n'a été trouvé dans la base de données.")
    
    # Enrichissement avec les tendances INSEE
    tendances = _prenoms_tendances.get(prenom, {})
    
    # Préparation d'un graphe simplifié pour les prénoms
    nodes, edges = [], []
    noms_groupe = data.get("prenoms_groupe", [])
    if not noms_groupe: noms_groupe = [data["prenom"]]
    
    for idx, g in enumerate(noms_groupe):
        is_main = (g == data["prenom"])
        # Fetch le count via tendances si dispo
        t_data = _prenoms_tendances.get(g, {})
        nb = t_data.get("total", 0)
        size = max(10, int(nb ** 0.5) / 10) + 5 if nb > 0 else 10
        nodes.append({
            "id": g, "label": g.title(), "value": size,
            "group": "principal" if is_main else "lie",
            "title": f"Naissances: {nb if nb else 'Inconnue'}"
        })
        if not is_main and idx != 0:
            edges.append({"from": data["prenom"], "to": g})
            
    graphe = {"nodes": nodes, "edges": edges}
    
    return render_template("fiche_prenom.html", fiche=data, tendances=tendances, graphe=graphe)


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip().lower()
    f_type = request.args.get("filter", "all").strip().lower()
    if len(q) < 2: return jsonify([])
    
    matches = []
    
    # Détection de recherche combinée
    parts = q.split()
    if len(parts) == 2 and f_type == "all":
        p1, p2 = parts[0], parts[1]
        # Cas 1: p1 = Prénom, p2 = Nom
        if p1 in INDEX_PRENOMS and p2 in INDEX_NOMS:
            matches.append({
                "type_entite": "compare",
                "id1": p1, "type1": "prenom", "nom_original_1": INDEX_PRENOMS[p1]["prenom_original"],
                "id2": p2, "type2": "nom", "nom_original_2": INDEX_NOMS[p2].get("nom_original", p2),
                "nom_original": f"Comparaison: {INDEX_PRENOMS[p1]['prenom_original']} & {INDEX_NOMS[p2].get('nom_original', p2)}"
            })
        # Cas 2: p1 = Nom, p2 = Prénom
        elif p1 in INDEX_NOMS and p2 in INDEX_PRENOMS:
            matches.append({
                "type_entite": "compare",
                "id1": p1, "type1": "nom", "nom_original_1": INDEX_NOMS[p1].get("nom_original", p1),
                "id2": p2, "type2": "prenom", "nom_original_2": INDEX_PRENOMS[p2]["prenom_original"],
                "nom_original": f"Comparaison: {INDEX_NOMS[p1].get('nom_original', p1)} & {INDEX_PRENOMS[p2]['prenom_original']}"
            })

    # Noms
    if f_type in ["all", "nom"]:
        for n in [n for n in NOMS_TRIES if n.startswith(q)][:10]:
            matches.append({
                "type_entite": "nom",
                "nom": INDEX_NOMS[n]["nom"], 
                "nom_original": INDEX_NOMS[n].get("nom_original", n) + " (Nom)"
            })
        
    # Prénoms
    if f_type in ["all", "prenom"]:
        for p in [p for p in PRENOMS_TRIES if p.startswith(q)][:10]:
            matches.append({
                "type_entite": "prenom",
                "nom": p, 
                "nom_original": INDEX_PRENOMS[p].get("prenom_original", p.title()) + " (Prénom)"
            })
        
    return jsonify(matches)


@app.route("/api/nom/<nom>")
def api_nom(nom):
    data = INDEX_NOMS.get(nom.strip().lower())
    if not data:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        **data,
        "carte":  preparer_carte(data),
        "chrono": extraire_date(data.get("origine_brute", "")),
        "graphe": preparer_graphe(data)
    })


@app.route("/api/prenom/<prenom>")
def api_prenom(prenom):
    prenom = prenom.strip().lower()
    data = INDEX_PRENOMS.get(prenom)
    if not data:
        return jsonify({"error": "not found"}), 404
        
    tendances = _prenoms_tendances.get(prenom, {})
    
    # Préparation d'un graphe simplifié pour les prénoms (comme dans la vue HTML)
    nodes, edges = [], []
    noms_groupe = data.get("prenoms_groupe", [])
    if not noms_groupe: noms_groupe = [data["prenom"]]
    
    for idx, g in enumerate(noms_groupe):
        is_main = (g == data["prenom"])
        t_data = _prenoms_tendances.get(g, {})
        nb = t_data.get("total", 0)
        size = max(10, int(nb ** 0.5) / 10) + 5 if nb > 0 else 10
        nodes.append({
            "id": g, "label": g.title(), "value": size,
            "group": "principal" if is_main else "lie",
            "title": f"Naissances: {nb if nb else 'Inconnue'}"
        })
        if not is_main and idx != 0:
            edges.append({"from": data["prenom"], "to": g})
            
    graphe = {"nodes": nodes, "edges": edges}

    return jsonify({
        **data,
        "tendances": tendances,
        "graphe": graphe
    })


@app.route("/compare")
def compare():
    id1 = request.args.get("id1", "").strip().lower()
    t1  = request.args.get("type1", "").strip().lower()
    id2 = request.args.get("id2", "").strip().lower()
    t2  = request.args.get("type2", "").strip().lower()
    
    data1, data2 = None, None
    stats1, stats2 = [], []
    
    def _get_chart_data(entity_id, entity_type):
        pts = []
        if entity_type == "nom":
            d = INDEX_NOMS.get(entity_id, {})
            insee = d.get("insee_data")
            if isinstance(insee, dict) and "historique" in insee:
                for period, count in insee["historique"].items():
                    try:
                        # "1891_1900" -> On place le point en 1900
                        y = int(period.replace("_", "-").split("-")[1])
                        # On divise par 10 pour avoir une moyenne annuelle comparable aux prénoms
                        pts.append({"x": y, "y": count / 10.0})
                    except: pass
        elif entity_type == "prenom":
            d = _prenoms_tendances.get(entity_id, {})
            if isinstance(d, dict) and "national" in d:
                for item in d["national"]:
                    pts.append({"x": item["annee"], "y": item["count"]})
        return sorted(pts, key=lambda p: p["x"])
        
    if id1 and t1 in ["nom", "prenom"]:
        data1 = INDEX_NOMS.get(id1) if t1 == "nom" else INDEX_PRENOMS.get(id1)
        if data1:
            data1 = dict(data1)
            data1["type_entite"] = t1
            stats1 = _get_chart_data(id1, t1)
            
    if id2 and t2 in ["nom", "prenom"]:
        data2 = INDEX_NOMS.get(id2) if t2 == "nom" else INDEX_PRENOMS.get(id2)
        if data2:
            data2 = dict(data2)
            data2["type_entite"] = t2
            stats2 = _get_chart_data(id2, t2)
            
    return render_template("compare.html", 
        data1=data1, t1=t1, stats1=stats1,
        data2=data2, t2=t2, stats2=stats2
    )


@app.route("/stats")
def stats():
    total_noms = len(INDEX_NOMS)
    total_prenoms = len(INDEX_PRENOMS)
    total_groupes_noms = len(INDEX_GROUPES)
    total_groupes_prenoms = len(_groupes_prenoms)
    
    # Agrégation par langue et par geographie
    lang_counter = defaultdict(int)
    geo_counter = defaultdict(int)
    yearly_dist = defaultdict(int)
    
    for n in _noms_final:
        if n.get("langue"): lang_counter[n["langue"]] += 1
        if n.get("geo"): geo_counter[n["geo"]] += 1
        
    for p in _prenoms_final:
        if p.get("langue"): lang_counter[p["langue"]] += 1
        if p.get("geo"): geo_counter[p["geo"]] += 1
        # Aggregation par année (chronologie)
        tend = _prenoms_tendances.get(p["prenom"], {})
        for entry in tend.get("national", []):
            yearly_dist[entry["annee"]] += entry["count"]

    # Top 10 langues et géo
    top_langs = sorted(lang_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    top_geos = sorted(geo_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Chronologie triée
    timeline = sorted([{"annee": a, "count": c} for a, c in yearly_dist.items() if a >= 1890], key=lambda x: x["annee"])

    # Top listes (triées par popularité totale)
    top_noms = []
    top_prenoms = []
    try:
        top_noms = sorted([n for n in _noms_final if "insee_data" in n and isinstance(n["insee_data"], dict)], 
                          key=lambda x: x["insee_data"].get("nombre_total", 0), reverse=True)[:10]
        
        top_prenoms_keys = sorted(_prenoms_tendances.keys(), 
                                  key=lambda k: _prenoms_tendances[k].get("total", 0), reverse=True)[:10]
        for k in top_prenoms_keys:
            if k in INDEX_PRENOMS:
                p_entry = dict(INDEX_PRENOMS[k])
                p_entry["total"] = _prenoms_tendances[k].get("total", 0)
                top_prenoms.append(p_entry)
    except Exception as e:
        print(f"Stats sorting error: {e}")
        
    # Prénoms "Tendance" (ceux dont le pic est récent, ex: > 2015)
    trending_prenoms = []
    try:
        trending_keys = []
        for p, d in _prenoms_tendances.items():
            if d.get("pic_annee", 0) >= 2018:
                trending_keys.append((p, d.get("total", 0)))
        
        # Sort by total popularity among those trending
        trending_keys = sorted(trending_keys, key=lambda x: x[1], reverse=True)[:10]
        for k, _ in trending_keys:
            if k in INDEX_PRENOMS:
                entry = dict(INDEX_PRENOMS[k])
                entry["pic_annee"] = _prenoms_tendances[k].get("pic_annee", "?")
                trending_prenoms.append(entry)
    except Exception as e:
        print(f"Trending calculation error: {e}")

    return render_template("stats.html", 
        total_noms=total_noms, 
        total_prenoms=total_prenoms,
        total_groupes_noms=total_groupes_noms,
        total_groupes_prenoms=total_groupes_prenoms,
        top_noms=top_noms,
        top_prenoms=top_prenoms,
        trending_prenoms=trending_prenoms,
        top_langs=top_langs,
        top_geos=top_geos,
        timeline=timeline
    )

@app.route("/admin/regroupement")
def admin_regroupement():
    # Renders the admin dashboard template
    return render_template("admin_regroupement.html")

@app.route("/api/admin/run_regroupement", methods=["POST"])
def api_admin_run_regroupement():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "error": "Requête JSON invalide ou manquante."}), 400
            
        cible = data.get("cible") # "noms" ou "prenoms"
        
        if cible not in ["noms", "prenoms"]:
            return jsonify({"status": "error", "error": "Cible invalide."}), 400

        # Find the best python executable
        venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "python.exe"))
        py_exec = venv_python if os.path.exists(venv_python) else sys.executable

        # Build the command line array based on the target script
        if cible == "prenoms":
            script_path = os.path.join("..", "prenoms", "2_regroupement_prenoms.py")
            cmd = [py_exec, script_path]
            
            # Add parameter overrides
            if "seuil_lev" in data: cmd.extend(["--seuil_lev", str(data["seuil_lev"])])
            if "prefixe_len" in data: cmd.extend(["--prefixe_len", str(data["prefixe_len"])])
            if "lev_k" in data: cmd.extend(["--lev_k", str(data["lev_k"])])
            if "seuil_sem" in data: cmd.extend(["--seuil_sem", str(data["seuil_sem"])])
            if "longueur_max" in data: cmd.extend(["--longueur_max", str(data["longueur_max"])])
            if "seuil_score_final" in data: cmd.extend(["--seuil_score_final", str(data["seuil_score_final"])])
            if "n_eval_paires" in data: cmd.extend(["--n_eval_paires", str(data["n_eval_paires"])])
                
        else: # noms
            script_path = os.path.join("..", "noms", "2_regroupement_noms.py")
            cmd = [py_exec, script_path]
            
            # Add parameter overrides
            if "seuil_lev_preselection" in data: cmd.extend(["--seuil_lev_preselection", str(data["seuil_lev_preselection"])])
            if "seuil_score_final" in data: cmd.extend(["--seuil_score_final", str(data["seuil_score_final"])])
            if "longueur_max" in data: cmd.extend(["--longueur_max", str(data["longueur_max"])])
            if "prefixe_len" in data: cmd.extend(["--prefixe_len", str(data["prefixe_len"])])
            if "lev_k" in data: cmd.extend(["--lev_k", str(data["lev_k"])])
            if "n_eval_paires" in data: cmd.extend(["--n_eval_paires", str(data["n_eval_paires"])])

        # Run the script in the parent directory context
        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cible))
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        
        # Parse standard output for metrics
        stdout = result.stderr + "\n" + result.stdout # Using stderr combined as logging output goes there by default
        
        # Basic parsing using regex to extract log lines
        import re
        stats = {
            "clusters": 0,
            "singletons": 0,
            "output": stdout
        }
        
        match = re.search(r"Clusters=(\d+) \| singletons=(\d+)", stdout)
        if match:
            stats["clusters"] = int(match.group(1))
            stats["singletons"] = int(match.group(2))
            
        return jsonify({"status": "success", "stats": stats})
        
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "error": f"Le script a échoué (Code {e.returncode})", "details": e.stderr if e.stderr else e.stdout}), 500
    except Exception as e:
        import traceback
        return jsonify({"status": "error", "error": str(e), "details": traceback.format_exc()}), 500

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.route("/admin/test_integration")
def admin_test_integration():
    return render_template("admin_test_integration.html")

# Global cache for model and index
_MODEL_CACHE = {"st": None, "faiss": None, "data_clean": None, "groupes_index": None}

@app.route("/api/admin/tester_integration", methods=["POST"])
def api_admin_tester_integration():
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        from rapidfuzz.distance import Levenshtein as Lev
    except ImportError:
        return jsonify({"status": "error", "error": "Packages 'sentence-transformers', 'faiss-cpu' ou 'rapidfuzz' manquants."}), 500

    data = request.json
    prenom = data.get("prenom", "").strip()
    texte = data.get("texte", "").strip()

    if not prenom or not texte:
        return jsonify({"status": "error", "error": "Prénom et texte requis."}), 400

    # 1. Normalisation & Analyse initiale
    def normaliser(s):
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = s.lower()
        s = re.sub(r"[-'']", " ", s)
        return " ".join(s.split())

    prenom_norm = normaliser(prenom)
    
    # Simulation d'extraction langue/géo (plus simple ici pour le test)
    langue = None
    if re.search(r"(?i)\borigine\s+arabe\b", texte): langue = "arabe"
    elif re.search(r"(?i)\borigine\s+grecque\b", texte): langue = "grec"
    elif re.search(r"(?i)\borigine\s+latine\b", texte): langue = "latin"
    elif re.search(r"(?i)\borigine\s+hébraïque\b", texte): langue = "hébreu"
    
    geo = None
    if re.search(r"(?i)\bfrance\b", texte): geo = "france"
    elif re.search(r"(?i)\ballemagne\b", texte): geo = "allemagne"

    # Liens Levenshtein (simplifié)
    liens_lev = []
    seuil = 1 if len(prenom_norm) <= 4 else (2 if len(prenom_norm) <= 7 else 3)
    for p_ref in PRENOMS_TRIES:
        if p_ref == prenom_norm: continue
        dist = Lev.distance(prenom_norm, p_ref, score_cutoff=seuil)
        if dist is not None and dist <= seuil:
            liens_lev.append((INDEX_PRENOMS[p_ref]["prenom_original"], int(dist)))
            if len(liens_lev) > 5: break

    # 2. Embedding & FAISS
    if _MODEL_CACHE["st"] is None:
        _MODEL_CACHE["st"] = SentenceTransformer("Lajavaness/sentence-camembert-base")
    
    texte_emb = texte + " | " + texte
    emb_query = _MODEL_CACHE["st"].encode([texte_emb], convert_to_numpy=True)
    faiss.normalize_L2(emb_query)

    if _MODEL_CACHE["faiss"] is None:
        path_npy = os.path.join(PRENOMS_DIR, "3_embeddings.npy")
        if os.path.exists(path_npy):
            embs = np.load(path_npy).astype('float32')
            idx = faiss.IndexFlatIP(embs.shape[1])
            idx.add(embs)
            _MODEL_CACHE["faiss"] = idx
            _MODEL_CACHE["total_vectors"] = embs.shape[0]
        else:
            return jsonify({"status": "error", "error": "Fichier d'embeddings non trouvé (3_embeddings.npy)."}), 500

    if _MODEL_CACHE["data_clean"] is None:
        _MODEL_CACHE["data_clean"] = _charger("2_prenoms_clean.json", PRENOMS_DIR)

    # Index des groupes : id_groupe_total -> liste des prénoms originaux du groupe
    if _MODEL_CACHE["groupes_index"] is None:
        groupes_raw = _charger("4_groupes_finals_prenoms.json", PRENOMS_DIR)
        # Chaque groupe contient une liste "prenoms" (normalisés) -> on cherche les noms originaux
        _grp_idx = {}
        for grp in groupes_raw:
            gid = grp.get("id_groupe_total", "")
            # Les prénoms dans le groupe sont en form normalisée ; on récupère les originaux via INDEX_PRENOMS
            membres_originaux = []
            for pn in grp.get("prenoms", []):
                if pn in INDEX_PRENOMS:
                    membres_originaux.append(INDEX_PRENOMS[pn]["prenom_original"])
                else:
                    membres_originaux.append(pn.title())
            _grp_idx[gid] = membres_originaux
        _MODEL_CACHE["groupes_index"] = _grp_idx

    D, I = _MODEL_CACHE["faiss"].search(emb_query, 5)
    
    matches = []
    for dist, idx in zip(D[0], I[0]):
        ref = _MODEL_CACHE["data_clean"][idx]
        sim_sem = float(dist)
        bonus_langue = 0.04 if langue and langue == ref.get("langue") else 0
        bonus_geo = 0.02 if geo and geo == ref.get("geo") else 0
        score = sim_sem + bonus_langue + bonus_geo
        
        # F1 score: precision = score final (inclus les bonuses), recall = sim sémantique brute
        # F1 = 2 * P * R / (P + R), plafonné à 1.0
        precision = min(score, 1.0)
        recall = sim_sem
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        # Prénoms liés au groupe du candidat
        gid = ref.get("id_groupe_total", "?")
        membres_groupe = _MODEL_CACHE["groupes_index"].get(gid, []) if _MODEL_CACHE["groupes_index"] else []
        # Exclure le candidat lui-même
        lies = [p for p in membres_groupe if p != ref["prenom_original"]][:8]

        matches.append({
            "nom": ref["prenom_original"],
            "id_groupe": gid,
            "lies": lies,
            "sim": sim_sem,
            "langue": ref.get("langue"),
            "geo": ref.get("geo"),
            "score": score,
            "f1": round(f1, 4),
            "decision": "FUSION" if score >= 0.92 else "REJET"
        })

    # Résumé global : F1 moyen & meilleur candidat
    best_match = max(matches, key=lambda m: m["f1"]) if matches else None
    global_f1 = round(sum(m["f1"] for m in matches) / len(matches), 4) if matches else 0.0

    return jsonify({
        "status": "success",
        "data": {
            "step1": {
                "prenom_norm": prenom_norm,
                "langue": langue,
                "geo": geo,
                "liens_lev": liens_lev
            },
            "step2": {
                "texte_emb": texte_emb[:200] + "...",
                "total_vectors": _MODEL_CACHE["total_vectors"]
            },
            "matches": matches,
            "summary": {
                "global_f1": global_f1,
                "best_candidate": best_match["nom"] if best_match else None,
                "best_f1": best_match["f1"] if best_match else None,
                "best_decision": best_match["decision"] if best_match else None,
            }
        }
    })


@app.route("/notebooks")
def list_notebooks():
    """Liste tous les notebooks (.ipynb) du projet de manière récursive."""
    notebooks = []
    
    # Parcourir récursivement pour trouver tous les notebooks
    for root, dirs, files in os.walk(BASE_DIR):
        # Exclure les dossiers inutiles
        if any(x in root for x in [".venv", ".git", ".ipynb_checkpoints", "node_modules"]):
            continue
            
        for file in files:
            if file.endswith(".ipynb"):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, BASE_DIR)
                
                notebooks.append({
                    "name": file,
                    "path": rel_path.replace("\\", "/")
                })
                
    # Trier par nom
    notebooks.sort(key=lambda x: x["name"])
    
    return render_template("notebooks.html", notebooks=notebooks)


@app.route("/notebook/view")
def view_notebook():
    """Convertit un notebook en HTML et l'affiche."""
    nb_path = request.args.get("path")
    if not nb_path:
        abort(404)
    
    # Sécurité : on restreint l'accès aux fichiers .ipynb uniquement
    abs_path = os.path.abspath(os.path.join(BASE_DIR, nb_path))
    if not abs_path.startswith(BASE_DIR) or not abs_path.endswith(".ipynb"):
        abort(403)
        
    if not os.path.isfile(abs_path):
        abort(404)
        
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            nb_node = nbformat.read(f, as_version=4)
            
        html_exporter = HTMLExporter()
        # On ne force pas le template 'classic' pour laisser nbconvert choisir le plus adapté
        (body, resources) = html_exporter.from_notebook_node(nb_node)
        
        # Injection des CSS Jupyter pour que le rendu soit correct (cellules, graphiques, etc.)
        custom_css = ""
        if 'inlining' in resources and 'css' in resources['inlining']:
            custom_css = "\n".join(resources['inlining']['css'])
        
        full_html = f"<style>{custom_css}</style>\n{body}"
        
        return render_template("notebook_view.html", notebook_html=full_html, notebook_name=os.path.basename(nb_path))
    except Exception as e:
        return f"Erreur lors de la conversion du notebook : {e}", 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)