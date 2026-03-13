"""
eval_summarize.py — Rapport d'évaluation Phase 3 (reformulation CamemBERT2CamemBERT)
=======================================================================================
Entrées :
    prenoms/data/3_prenoms_grouped.json   — textes sources (avant reformulation)
    prenoms/data/4_prenoms_final.json     — textes reformulés (sortie Phase 3)
    prenoms/data/4_cache_paraphrase.pkl   — cache (métriques de run)
    prenoms/data/3_eval_prenoms.json      — paires Phase 2 (couverture groupes)

Sortie :
    prenoms/eval/rapport_phase3.html      — rapport HTML self-contained

Métriques calculées (dans l'ordre de priorité) :

    1. PERFORMANCE
        - Taux de cache hit (textes servis sans appel modèle)
        - Distribution des longueurs de texte (tokens) par champ
        - Temps estimé économisé par le cache

    2. QUALITE LINGUISTIQUE
        - Score de lisibilité Flesch–Kincaid adapté au français (longueur phrases,
          longueur mots) — proxy rapide sans annotation humaine
        - Taux de phrases bien formées (majuscule + ponctuation finale)
        - Détection de répétitions de trigrammes (symptôme du mode greedy)

    3. DIVERSIFICATION
        - Sim cosine TF-IDF source vs reformulé (distribution + seuil DEDUP)
        - ROUGE-1/ROUGE-2/ROUGE-L source vs reformulé (chevauchement lexical)
        - Ratio longueur reformulé/source (compression moyenne par champ)
        - Taux de repli sur source (sim > DEDUP_SEUIL)

    4. COUVERTURE
        - Taux de champs renseignés par champ
        - Distribution des tailles de groupes

Dépendances :
    pip install scikit-learn numpy rouge-score tqdm
    (rouge-score pour ROUGE, pas de dépendance lourde)
"""

import hashlib
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge-score non installe — ROUGE desactive. pip install rouge-score")

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

INPUT_SOURCE   = "prenoms/data/3_prenoms_grouped.json"
INPUT_FINAL    = "prenoms/data/4_prenoms_final.json"
INPUT_CACHE    = "prenoms/data/4_cache_paraphrase.pkl"
INPUT_EVAL_P2  = "prenoms/data/3_eval_prenoms.json"
OUTPUT_RAPPORT = "prenoms/eval/rapport_phase3.html"

CHAMPS = ("etymologie", "provenance", "histoire", "signification")
DEDUP_SEUIL = 0.97

# Estimation du temps CPU moyen par texte avec num_beams=1 (secondes).
# Calibré empiriquement sur le pipeline — ajuster si le run réel diffère.
TEMPS_PAR_TEXTE_S = 1.5


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def charger_donnees():
    """
    Charge et aligne source/final par id prénom.
    Retourne (source_map, final_map, cache, eval_p2).
    source_map / final_map : dict[id -> dict]
    """
    with open(INPUT_SOURCE, "r", encoding="utf-8") as f:
        source_list = json.load(f)
    with open(INPUT_FINAL, "r", encoding="utf-8") as f:
        final_list = json.load(f)

    source_map = {d["id"]: d for d in source_list}
    final_map  = {d["id"]: d for d in final_list}

    cache = {}
    if os.path.isfile(INPUT_CACHE):
        with open(INPUT_CACHE, "rb") as f:
            cache = pickle.load(f)

    eval_p2 = {}
    if os.path.isfile(INPUT_EVAL_P2):
        with open(INPUT_EVAL_P2, "r", encoding="utf-8") as f:
            eval_p2 = json.load(f)

    print(f"Source : {len(source_map)} prenoms | Final : {len(final_map)} prenoms")
    print(f"Cache  : {len(cache)} entrees")
    return source_map, final_map, cache, eval_p2


# ---------------------------------------------------------------------------
# Métriques Performance
# ---------------------------------------------------------------------------

def metriques_performance(source_map: dict, cache: dict) -> dict:
    """
    Calcule le taux de cache hit et le temps estimé économisé.

    Cache hit : un texte est servi depuis le cache si sa clé SHA256 est
    présente avant le run. On ne peut pas distinguer les entrées pré-existantes
    des nouvelles sans journal de run — on utilise le cache final comme proxy
    du total des textes uniques traités.
    """
    # Inventaire des clés SHA256 de tous les textes sources
    cles_sources: set = set()
    for item in source_map.values():
        for champ in CHAMPS:
            t = item.get(champ) or ""
            if t.strip():
                cles_sources.add(hashlib.sha256(t.encode()).hexdigest())

    total_unique  = len(cles_sources)
    total_cache   = len(cache)
    # Hit = clés sources présentes dans le cache (textes servis sans appel modèle)
    hits          = len(cles_sources & set(cache.keys()))
    taux_hit      = hits / total_unique if total_unique else 0.0

    # Toutes les occurrences de champs non vides (textes total, avec doublons)
    n_appels_bruts = sum(
        1 for item in source_map.values()
        for champ in CHAMPS
        if (item.get(champ) or "").strip()
    )
    # Appels modèle évités = occurrences dont la clé est en cache
    cles_par_occurrence = [
        hashlib.sha256((item.get(champ) or "").encode()).hexdigest()
        for item in source_map.values()
        for champ in CHAMPS
        if (item.get(champ) or "").strip()
    ]
    appels_evites = sum(1 for c in cles_par_occurrence if c in cache)
    temps_economise_h = appels_evites * TEMPS_PAR_TEXTE_S / 3600

    # Distribution longueurs (en chars) par champ
    longueurs_par_champ = {champ: [] for champ in CHAMPS}
    for item in source_map.values():
        for champ in CHAMPS:
            t = item.get(champ) or ""
            if t.strip():
                longueurs_par_champ[champ].append(len(t))

    stats_longueurs = {}
    for champ, llist in longueurs_par_champ.items():
        if llist:
            arr = np.array(llist)
            stats_longueurs[champ] = {
                "n":       len(arr),
                "moy":     float(np.mean(arr)),
                "med":     float(np.median(arr)),
                "p90":     float(np.percentile(arr, 90)),
                "max":     int(np.max(arr)),
                "pct_long": float(np.mean(arr > 1800) * 100),  # > MAX_CHARS
            }

    return {
        "total_unique_textes": total_unique,
        "total_cache_entrees": total_cache,
        "cache_hits":          hits,
        "taux_hit_pct":        round(taux_hit * 100, 1),
        "n_appels_bruts":      n_appels_bruts,
        "appels_evites":       appels_evites,
        "temps_economise_h":   round(temps_economise_h, 2),
        "stats_longueurs":     stats_longueurs,
    }


# ---------------------------------------------------------------------------
# Métriques Qualité linguistique
# ---------------------------------------------------------------------------

def score_flesch_fr(texte: str) -> float:
    """
    Adaptation Flesch–Kincaid pour le français.
    Formule Kandel–Moles (1958) — la seule calibrée sur le français :
        FK_fr = 207 - 1.015 * (mots/phrases) - 73.6 * (syllabes/mots)
    Approximation syllabes : comptage des voyelles (a,e,i,o,u,y + accents).
    Score > 65 = lisible, 40-65 = moyen, < 40 = difficile.
    Retourne NaN si le texte est trop court pour être significatif.
    """
    phrases = [p.strip() for p in re.split(r"[.!?]+", texte) if p.strip()]
    mots    = re.findall(r"\b\w+\b", texte)
    if len(phrases) < 1 or len(mots) < 3:
        return float("nan")

    n_syllabes = sum(
        len(re.findall(r"[aeiouyàâäéèêëïîôùûüœæ]", m.lower()))
        for m in mots
    )
    mots_par_phrase = len(mots) / len(phrases)
    syllabes_par_mot = n_syllabes / len(mots) if mots else 1

    return 207.0 - 1.015 * mots_par_phrase - 73.6 * syllabes_par_mot


def taux_bien_forme(textes: list) -> float:
    """
    Proportion de textes avec majuscule initiale ET ponctuation finale.
    Indicateur de qualité de `nettoyer_reformulation`.
    """
    if not textes:
        return 0.0
    ok = sum(
        1 for t in textes
        if t and t[0].isupper() and t[-1] in ".!?"
    )
    return ok / len(textes)


def taux_repetitions_trigrammes(texte: str, seuil_rep: int = 3) -> float:
    """
    Proportion de trigrammes de mots apparaissant >= seuil_rep fois.
    Signal du mode greedy (num_beams=1) qui répète des séquences.
    Valeur > 0.05 indique un problème de répétition.
    """
    mots = re.findall(r"\b\w+\b", texte.lower())
    if len(mots) < 3:
        return 0.0
    trigrams = [tuple(mots[i:i+3]) for i in range(len(mots) - 2)]
    counts   = Counter(trigrams)
    repetes  = sum(1 for c in counts.values() if c >= seuil_rep)
    return repetes / len(trigrams) if trigrams else 0.0


def metriques_qualite(final_map: dict) -> dict:
    """
    Calcule Flesch-FR, taux bien formé, taux répétitions sur les textes reformulés.
    """
    resultats = {champ: {"flesch": [], "bien_forme": [], "rep_tg": []} for champ in CHAMPS}

    for item in final_map.values():
        for champ in CHAMPS:
            t = item.get(champ) or ""
            if not t.strip():
                continue
            fk = score_flesch_fr(t)
            if not math.isnan(fk):
                resultats[champ]["flesch"].append(fk)
            resultats[champ]["bien_forme"].append(1 if (t[0].isupper() and t[-1] in ".!?") else 0)
            resultats[champ]["rep_tg"].append(taux_repetitions_trigrammes(t))

    stats = {}
    for champ in CHAMPS:
        fl  = resultats[champ]["flesch"]
        bf  = resultats[champ]["bien_forme"]
        rep = resultats[champ]["rep_tg"]
        stats[champ] = {
            "flesch_moy":       round(float(np.mean(fl)),  2) if fl  else None,
            "flesch_med":       round(float(np.median(fl)),2) if fl  else None,
            "taux_bien_forme":  round(float(np.mean(bf)) * 100, 1) if bf  else None,
            "taux_rep_tg_moy":  round(float(np.mean(rep))* 100, 2) if rep else None,
            "taux_rep_tg_p90":  round(float(np.percentile(rep, 90)) * 100, 2) if rep else None,
        }
    return stats


# ---------------------------------------------------------------------------
# Métriques Diversification
# ---------------------------------------------------------------------------

def metriques_diversification(source_map: dict, final_map: dict) -> dict:
    """
    Pour chaque paire (source, reformulé) alignée par id+champ :
        - Sim cosine TF-IDF
        - ROUGE-1/2/L recall (chevauchement lexical source→reformulé)
        - Ratio de longueur reformulé/source
        - Flag repli (sim > DEDUP_SEUIL)

    TF-IDF entraîné sur l'union source+reformulé pour un espace commun.
    ROUGE recall mesure la rétention d'information (combien du source est
    conservé dans le reformulé) — valeur < 0.6 indique une sur-compression.
    """
    ids_communs = set(source_map) & set(final_map)

    # Construction du corpus pour TF-IDF
    corpus_tfidf = []
    paires_ids   = []
    for pid in ids_communs:
        for champ in CHAMPS:
            src = (source_map[pid].get(champ) or "").strip()
            ref = (final_map[pid].get(champ)  or "").strip()
            if src and ref:
                corpus_tfidf.append(src)
                corpus_tfidf.append(ref)
                paires_ids.append((pid, champ, src, ref))

    if not corpus_tfidf:
        return {}

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    vec.fit(corpus_tfidf)

    scorer_rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    ) if HAS_ROUGE else None

    resultats = {champ: {
        "sims": [], "rouge1": [], "rouge2": [], "rougeL": [],
        "ratio_len": [], "n_repli": 0, "n_total": 0,
    } for champ in CHAMPS}

    for pid, champ, src, ref in tqdm(paires_ids, desc="Diversification", unit="paire"):
        r = resultats[champ]
        r["n_total"] += 1

        # Sim cosine TF-IDF
        vecs = vec.transform([src, ref])
        sim  = float(cosine_similarity(vecs[0], vecs[1])[0, 0])
        r["sims"].append(sim)
        if sim > DEDUP_SEUIL:
            r["n_repli"] += 1

        # Ratio longueur
        r["ratio_len"].append(len(ref) / len(src) if src else 1.0)

        # ROUGE
        if scorer_rouge:
            scores = scorer_rouge.score(src, ref)
            r["rouge1"].append(scores["rouge1"].recall)
            r["rouge2"].append(scores["rouge2"].recall)
            r["rougeL"].append(scores["rougeL"].recall)

    stats = {}
    for champ in CHAMPS:
        r = resultats[champ]
        def _stat(lst):
            if not lst:
                return None
            a = np.array(lst)
            return {
                "moy": round(float(np.mean(a)), 4),
                "med": round(float(np.median(a)), 4),
                "p10": round(float(np.percentile(a, 10)), 4),
                "p90": round(float(np.percentile(a, 90)), 4),
                # distribution par intervalles pour histogramme
                "hist": np.histogram(a, bins=10, range=(0.0, 1.0))[0].tolist(),
            }
        stats[champ] = {
            "sim_cosine":  _stat(r["sims"]),
            "rouge1":      _stat(r["rouge1"])      if HAS_ROUGE else None,
            "rouge2":      _stat(r["rouge2"])      if HAS_ROUGE else None,
            "rougeL":      _stat(r["rougeL"])      if HAS_ROUGE else None,
            "ratio_len":   _stat(r["ratio_len"]),
            "taux_repli":  round(r["n_repli"] / r["n_total"] * 100, 1) if r["n_total"] else 0,
            "n_total":     r["n_total"],
            "n_repli":     r["n_repli"],
        }
    return stats


# ---------------------------------------------------------------------------
# Métriques Couverture
# ---------------------------------------------------------------------------

def metriques_couverture(source_map: dict, final_map: dict, eval_p2: dict) -> dict:
    """
    Taux de champs renseignés source vs reformulé + distribution groupes Phase 2.
    """
    couverture = {}
    for champ in CHAMPS:
        n_src = sum(1 for d in source_map.values() if (d.get(champ) or "").strip())
        n_ref = sum(1 for d in final_map.values()  if (d.get(champ) or "").strip())
        n_tot = len(source_map)
        couverture[champ] = {
            "source_pct": round(n_src / n_tot * 100, 1) if n_tot else 0,
            "final_pct":  round(n_ref / n_tot * 100, 1) if n_tot else 0,
            "n_source":   n_src,
            "n_final":    n_ref,
        }

    # Distribution tailles groupes depuis final
    tailles_groupes = Counter()
    for item in final_map.values():
        g = item.get("prenoms_groupe", [])
        tailles_groupes[len(g)] += 1

    # Outliers Phase 2 depuis eval_p2
    n_outliers = 0
    if isinstance(eval_p2, dict) and "outliers" in eval_p2:
        n_outliers = sum(len(o.get("outliers", [])) for o in eval_p2["outliers"])

    return {
        "par_champ":        couverture,
        "tailles_groupes":  dict(sorted(tailles_groupes.items())),
        "n_prenoms":        len(final_map),
        "n_groupes":        len({d.get("id_groupe_total") for d in final_map.values()}),
        "n_outliers_p2":    n_outliers,
    }


# ---------------------------------------------------------------------------
# Échantillons qualitatifs (side-by-side)
# ---------------------------------------------------------------------------

def echantillons_comparatifs(source_map: dict, final_map: dict,
                              n_par_champ: int = 5) -> dict:
    """
    Sélectionne n_par_champ exemples par champ :
        - 2 avec sim cosine basse (bonne diversification)
        - 2 avec sim cosine haute (repli probable)
        - 1 aléatoire

    Retourne dict[champ -> list[{source, reformule, sim}]]
    """
    import random
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1)

    ids_communs = sorted(set(source_map) & set(final_map))
    echantillons = {}

    for champ in CHAMPS:
        paires = []
        for pid in ids_communs:
            src = (source_map[pid].get(champ) or "").strip()
            ref = (final_map[pid].get(champ)  or "").strip()
            if src and ref and len(src) > 30:
                paires.append((pid, src, ref))

        if not paires:
            echantillons[champ] = []
            continue

        corpus = [s for _, s, _ in paires] + [r for _, _, r in paires]
        try:
            vec_local = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1)
            vec_local.fit(corpus)
            sims = []
            for pid, src, ref in paires:
                v = vec_local.transform([src, ref])
                sims.append(float(cosine_similarity(v[0], v[1])[0,0]))
        except Exception:
            sims = [0.5] * len(paires)

        paires_sim = sorted(zip(sims, paires), key=lambda x: x[0])
        selection  = []

        # 2 basse sim, 2 haute sim, 1 aléatoire (sans doublon)
        vus = set()
        for sim, (pid, src, ref) in paires_sim[:3]:
            if pid not in vus:
                selection.append({"prenom": pid, "source": src, "reformule": ref,
                                   "sim": round(sim, 4), "tag": "basse_sim"})
                vus.add(pid)
        for sim, (pid, src, ref) in reversed(paires_sim[-3:]):
            if pid not in vus:
                selection.append({"prenom": pid, "source": src, "reformule": ref,
                                   "sim": round(sim, 4), "tag": "haute_sim"})
                vus.add(pid)
        candidats_aleat = [(s, p) for s, p in zip(sims, paires) if p[0] not in vus]
        if candidats_aleat:
            sim, (pid, src, ref) = random.choice(candidats_aleat)
            selection.append({"prenom": pid, "source": src, "reformule": ref,
                               "sim": round(sim, 4), "tag": "aleatoire"})

        echantillons[champ] = selection[:n_par_champ]

    return echantillons


# ---------------------------------------------------------------------------
# Génération HTML
# ---------------------------------------------------------------------------

_CSS = """
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
    --accent: #6c8ebf; --accent2: #5cb85c; --warn: #f0ad4e; --danger: #d9534f;
    --text: #e8eaf0; --muted: #8891a8; --border: #2e3350;
    --font: 'Inter', system-ui, sans-serif; --mono: 'JetBrains Mono', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         font-size: 14px; line-height: 1.6; }
  .container { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 22px; font-weight: 700; color: var(--accent); margin-bottom: 4px; }
  h2 { font-size: 16px; font-weight: 600; color: var(--text); margin: 32px 0 16px;
       border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  h3 { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase;
       letter-spacing: .05em; margin: 20px 0 10px; }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 28px; }
  .grid { display: grid; gap: 16px; }
  .grid-2 { grid-template-columns: 1fr 1fr; }
  .grid-4 { grid-template-columns: repeat(4, 1fr); }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 10px; padding: 18px; }
  .kpi { text-align: center; }
  .kpi .val { font-size: 32px; font-weight: 700; color: var(--accent); }
  .kpi .lbl { font-size: 11px; color: var(--muted); text-transform: uppercase;
              letter-spacing: .06em; margin-top: 4px; }
  .kpi.good .val  { color: var(--accent2); }
  .kpi.warn .val  { color: var(--warn); }
  .kpi.bad  .val  { color: var(--danger); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: var(--surface2); color: var(--muted); font-weight: 600;
       font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
       padding: 8px 12px; text-align: left; }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 99px;
           font-size: 11px; font-weight: 600; }
  .badge-good { background: #1d3a1d; color: var(--accent2); }
  .badge-warn { background: #3a2e1d; color: var(--warn); }
  .badge-bad  { background: #3a1d1d; color: var(--danger); }
  .bar-wrap { background: var(--surface2); border-radius: 4px; height: 8px;
              overflow: hidden; width: 100%; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width .3s; }
  .hist-wrap { display: flex; align-items: flex-end; gap: 2px; height: 50px; margin-top: 6px; }
  .hist-bar { background: var(--accent); opacity: .7; flex: 1; border-radius: 2px 2px 0 0;
              min-height: 2px; }
  .compare-pair { border: 1px solid var(--border); border-radius: 8px;
                  margin-bottom: 12px; overflow: hidden; }
  .compare-header { background: var(--surface2); padding: 8px 14px; font-size: 11px;
                    color: var(--muted); display: flex; justify-content: space-between; }
  .compare-body { display: grid; grid-template-columns: 1fr 1fr; }
  .compare-col { padding: 12px 14px; font-size: 12px; line-height: 1.7; }
  .compare-col:first-child { border-right: 1px solid var(--border); }
  .compare-col-header { font-size: 10px; font-weight: 700; text-transform: uppercase;
                        letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; }
  .tabs { display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }
  .tab { padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px;
         font-weight: 600; border: 1px solid var(--border); background: var(--surface);
         color: var(--muted); transition: all .15s; }
  .tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .section-divider { height: 1px; background: var(--border); margin: 40px 0; }
  .footnote { font-size: 11px; color: var(--muted); margin-top: 8px; }
  canvas { max-width: 100%; }
</style>
"""

_JS_TABS = """
<script>
function showTab(groupId, tabId) {
  document.querySelectorAll('[data-group="' + groupId + '"].tab-panel')
    .forEach(p => p.classList.remove('active'));
  document.querySelectorAll('[data-group="' + groupId + '"].tab')
    .forEach(t => t.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  document.querySelector('[data-tab="' + tabId + '"]').classList.add('active');
}
</script>
"""


def _badge(val, seuils_bon_mauvais, fmt="{:.1f}"):
    """Retourne un badge HTML coloré selon les seuils (bon >= s1, mauvais < s2)."""
    s_bon, s_mauv = seuils_bon_mauvais
    cls = "badge-good" if val >= s_bon else ("badge-bad" if val < s_mauv else "badge-warn")
    return f'<span class="badge {cls}">{fmt.format(val)}</span>'


def _hist_html(hist_counts: list) -> str:
    if not hist_counts:
        return ""
    mx = max(hist_counts) or 1
    bars = "".join(
        f'<div class="hist-bar" style="height:{c/mx*100:.0f}%" title="{c}"></div>'
        for c in hist_counts
    )
    return f'<div class="hist-wrap">{bars}</div>'


def _bar_html(pct: float, color: str = "var(--accent)") -> str:
    return (
        f'<div class="bar-wrap"><div class="bar-fill" '
        f'style="width:{min(pct,100):.1f}%;background:{color}"></div></div>'
    )


def generer_html(perf: dict, qualite: dict, divers: dict,
                 couv: dict, echant: dict, version_info: str) -> str:

    # ------------------------------------------------------------------
    # Section Performance
    # ------------------------------------------------------------------
    kpi_hit_cls = "good" if perf["taux_hit_pct"] >= 80 else ("warn" if perf["taux_hit_pct"] >= 40 else "bad")
    sec_perf = f"""
<h2>1. Performance</h2>
<div class="grid grid-4" style="margin-bottom:20px">
  <div class="card kpi {kpi_hit_cls}">
    <div class="val">{perf['taux_hit_pct']}%</div>
    <div class="lbl">Cache hit rate</div>
  </div>
  <div class="card kpi">
    <div class="val">{perf['total_cache_entrees']:,}</div>
    <div class="lbl">Entrees cache</div>
  </div>
  <div class="card kpi">
    <div class="val">{perf['appels_evites']:,}</div>
    <div class="lbl">Appels modele evites</div>
  </div>
  <div class="card kpi good">
    <div class="val">{perf['temps_economise_h']:.1f}h</div>
    <div class="lbl">Temps economise (est.)</div>
  </div>
</div>
<div class="card">
<h3>Distribution des longueurs de texte source (chars) par champ</h3>
<table>
<thead><tr>
  <th>Champ</th><th>N</th><th>Moyenne</th><th>Mediane</th>
  <th>P90</th><th>Max</th><th>% > 1800 chars</th>
</tr></thead><tbody>"""
    for champ, s in perf["stats_longueurs"].items():
        warn_long = s["pct_long"] > 5
        sec_perf += f"""
<tr>
  <td><strong>{champ}</strong></td>
  <td>{s['n']:,}</td>
  <td>{s['moy']:.0f}</td>
  <td>{s['med']:.0f}</td>
  <td>{s['p90']:.0f}</td>
  <td>{s['max']}</td>
  <td>{'<span class="badge badge-warn">' if warn_long else ''}{s['pct_long']:.1f}%{'</span>' if warn_long else ''}</td>
</tr>"""
    sec_perf += """</tbody></table>
<p class="footnote">Textes > 1800 chars : segmentes en blocs par _segmenter() avant generation.</p>
</div>"""

    # ------------------------------------------------------------------
    # Section Qualité linguistique
    # ------------------------------------------------------------------
    sec_qual = """
<h2>2. Qualite linguistique</h2>
<div class="card">
<table>
<thead><tr>
  <th>Champ</th>
  <th>Flesch-FR moy</th><th>Flesch-FR med</th>
  <th>Bien forme %</th>
  <th>Repetitions tg moy %</th><th>Repetitions tg P90 %</th>
</tr></thead><tbody>"""
    for champ in CHAMPS:
        q = qualite.get(champ, {})
        fk_m = q.get("flesch_moy")
        fk_med = q.get("flesch_med")
        bf    = q.get("taux_bien_forme")
        rep_m = q.get("taux_rep_tg_moy")
        rep90 = q.get("taux_rep_tg_p90")
        fk_badge  = _badge(fk_m,   (65, 40), "{:.1f}") if fk_m  is not None else "—"
        bf_badge  = _badge(bf,     (95, 80), "{:.1f}%") if bf    is not None else "—"
        rep_badge = _badge(rep_m,  (0, 5), "{:.2f}%") if rep_m is not None else "—"
        # Pour rep : inversé — 0% est bon, >5% est mauvais
        rep_badge = (f'<span class="badge badge-good">{rep_m:.2f}%</span>'
                     if rep_m is not None and rep_m < 2
                     else (f'<span class="badge badge-bad">{rep_m:.2f}%</span>'
                           if rep_m is not None and rep_m >= 5
                           else (f'<span class="badge badge-warn">{rep_m:.2f}%</span>'
                                 if rep_m is not None else "—")))
        sec_qual += f"""
<tr>
  <td><strong>{champ}</strong></td>
  <td>{fk_badge}</td>
  <td>{f"{fk_med:.1f}" if fk_med is not None else "—"}</td>
  <td>{bf_badge}</td>
  <td>{rep_badge}</td>
  <td>{f"{rep90:.2f}%" if rep90 is not None else "—"}</td>
</tr>"""
    sec_qual += """</tbody></table>
<p class="footnote">
  Flesch-FR : formule Kandel-Moles — &gt;65 lisible, 40-65 moyen, &lt;40 difficile.<br>
  Bien forme : majuscule initiale + ponctuation finale (.!?).<br>
  Repetitions trigrammes : taux de trigrammes apparaissant &ge;3 fois — signal du mode greedy.
</p>
</div>"""

    # ------------------------------------------------------------------
    # Section Diversification
    # ------------------------------------------------------------------
    sec_div = """<h2>3. Diversification</h2>"""
    if not divers:
        sec_div += '<div class="card"><p style="color:var(--muted)">Aucune donnee disponible.</p></div>'
    else:
        sec_div += """<div class="card"><table>
<thead><tr>
  <th>Champ</th>
  <th>Sim cosine moy</th><th>Distribution sim</th>
  <th>ROUGE-1 recall</th><th>ROUGE-L recall</th>
  <th>Ratio longueur</th><th>Taux repli</th><th>N paires</th>
</tr></thead><tbody>"""
        for champ in CHAMPS:
            d = divers.get(champ, {})
            sc = d.get("sim_cosine", {}) or {}
            r1 = d.get("rouge1",    {}) or {}
            rl = d.get("rougeL",    {}) or {}
            ra = d.get("ratio_len", {}) or {}
            tr = d.get("taux_repli", 0)
            nt = d.get("n_total", 0)

            sim_v  = sc.get("moy")
            r1_v   = r1.get("moy")
            rl_v   = rl.get("moy")
            ra_v   = ra.get("moy")

            sim_badge = (
                f'<span class="badge badge-good">{sim_v:.3f}</span>'
                if sim_v is not None and sim_v < 0.7 else
                f'<span class="badge badge-warn">{sim_v:.3f}</span>'
                if sim_v is not None and sim_v < DEDUP_SEUIL else
                f'<span class="badge badge-bad">{sim_v:.3f}</span>'
                if sim_v is not None else "—"
            )
            repli_badge = (
                f'<span class="badge badge-good">{tr:.1f}%</span>' if tr < 10 else
                f'<span class="badge badge-warn">{tr:.1f}%</span>' if tr < 30 else
                f'<span class="badge badge-bad">{tr:.1f}%</span>'
            )
            hist_html = _hist_html(sc.get("hist", []))

            sec_div += f"""
<tr>
  <td><strong>{champ}</strong></td>
  <td>{sim_badge}</td>
  <td style="min-width:100px">{hist_html}<div style="font-size:10px;color:var(--muted);margin-top:2px">0 → 1</div></td>
  <td>{f"{r1_v:.3f}" if r1_v is not None else "n/a"}</td>
  <td>{f"{rl_v:.3f}" if rl_v is not None else "n/a"}</td>
  <td>{f"{ra_v:.2f}" if ra_v is not None else "—"}</td>
  <td>{repli_badge}</td>
  <td>{nt:,}</td>
</tr>"""
        sec_div += """</tbody></table>
<p class="footnote">
  Sim cosine TF-IDF : &lt;0.70 = bonne diversification, &gt;0.97 = repli sur source.<br>
  ROUGE recall : proportion du source conservée dans le reformulé — &lt;0.6 indique sur-compression.<br>
  Ratio longueur : reformulé / source — &lt;1.0 = compression, &gt;1.0 = expansion.
</p></div>"""

    # ------------------------------------------------------------------
    # Section Couverture
    # ------------------------------------------------------------------
    sec_couv = f"""
<h2>4. Couverture</h2>
<div class="grid grid-2">
<div class="card">
<h3>Taux de renseignement par champ</h3>
<table>
<thead><tr><th>Champ</th><th>Source %</th><th>Reformule %</th><th>Delta</th></tr></thead>
<tbody>"""
    for champ in CHAMPS:
        c = couv["par_champ"].get(champ, {})
        sp = c.get("source_pct", 0)
        fp = c.get("final_pct", 0)
        delta = fp - sp
        dcol = "var(--accent2)" if delta >= 0 else "var(--danger)"
        sec_couv += f"""
<tr>
  <td><strong>{champ}</strong></td>
  <td>{sp:.1f}%</td>
  <td>{fp:.1f}%</td>
  <td style="color:{dcol}">{delta:+.1f}%</td>
</tr>"""
    sec_couv += f"""</tbody></table></div>
<div class="card">
<h3>Synthese</h3>
<table>
<tbody>
<tr><td>Prenoms total</td><td><strong>{couv['n_prenoms']:,}</strong></td></tr>
<tr><td>Groupes total</td><td><strong>{couv['n_groupes']:,}</strong></td></tr>
<tr><td>Outliers Phase 2</td><td><strong>{couv['n_outliers_p2']:,}</strong></td></tr>
</tbody></table>
<h3 style="margin-top:16px">Distribution tailles groupes</h3>
<table><thead><tr><th>Taille</th><th>N groupes</th><th>Proportion</th></tr></thead><tbody>"""
    total_grp = sum(couv["tailles_groupes"].values())
    for taille, nb in sorted(couv["tailles_groupes"].items())[:15]:
        pct = nb / total_grp * 100 if total_grp else 0
        sec_couv += f"""
<tr>
  <td>{taille}</td>
  <td>{nb}</td>
  <td>{_bar_html(pct)} {pct:.1f}%</td>
</tr>"""
    if len(couv["tailles_groupes"]) > 15:
        sec_couv += f"<tr><td colspan='3' style='color:var(--muted);font-style:italic'>... {len(couv['tailles_groupes'])-15} autres tailles</td></tr>"
    sec_couv += "</tbody></table></div></div>"

    # ------------------------------------------------------------------
    # Section Comparatifs qualitatifs (tabs par champ)
    # ------------------------------------------------------------------
    sec_comp = """<h2>5. Comparatifs source / reformule</h2>"""
    tabs_html  = '<div class="tabs">'
    panels_html = ""
    for idx, champ in enumerate(CHAMPS):
        active = "active" if idx == 0 else ""
        tab_id = f"tab_comp_{champ}"
        tabs_html += (
            f'<div class="tab {active}" data-group="comp" data-tab="{tab_id}" '
            f'onclick="showTab(\'comp\',\'{tab_id}\')">{champ}</div>'
        )
        panels_html += f'<div id="{tab_id}" class="tab-panel {active}" data-group="comp">'
        for ex in echant.get(champ, []):
            tag_lbl = {"basse_sim": "Bonne diversification",
                       "haute_sim": "Repli probable",
                       "aleatoire": "Aleatoire"}.get(ex["tag"], ex["tag"])
            tag_cls = {"basse_sim": "badge-good",
                       "haute_sim": "badge-bad",
                       "aleatoire": ""}.get(ex["tag"], "")
            panels_html += f"""
<div class="compare-pair">
  <div class="compare-header">
    <span><strong>{ex['prenom']}</strong> — {champ}</span>
    <span>
      <span class="badge {tag_cls}">{tag_lbl}</span>
      &nbsp; sim cosine = {ex['sim']}
    </span>
  </div>
  <div class="compare-body">
    <div class="compare-col">
      <div class="compare-col-header">Source</div>
      {ex['source']}
    </div>
    <div class="compare-col">
      <div class="compare-col-header">Reformule</div>
      {ex['reformule']}
    </div>
  </div>
</div>"""
        panels_html += "</div>"
    tabs_html += "</div>"
    sec_comp  += tabs_html + panels_html

    # ------------------------------------------------------------------
    # Assemblage final
    # ------------------------------------------------------------------
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rapport Phase 3 — Reformulation prénoms</title>
{_CSS}
</head>
<body>
<div class="container">
  <h1>Rapport d'evaluation — Phase 3 Reformulation</h1>
  <p class="meta">{version_info} &nbsp;|&nbsp; ROUGE : {"actif" if HAS_ROUGE else "inactif (pip install rouge-score)"}</p>
  {sec_perf}
  <div class="section-divider"></div>
  {sec_qual}
  <div class="section-divider"></div>
  {sec_div}
  <div class="section-divider"></div>
  {sec_couv}
  <div class="section-divider"></div>
  {sec_comp}
</div>
{_JS_TABS}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    import datetime

    for chemin in (INPUT_SOURCE, INPUT_FINAL):
        if not os.path.isfile(chemin):
            print(f"[ERREUR] Fichier introuvable : {chemin}")
            return

    os.makedirs(os.path.dirname(OUTPUT_RAPPORT), exist_ok=True)

    print("Chargement des donnees...")
    source_map, final_map, cache, eval_p2 = charger_donnees()

    print("Calcul metriques performance...")
    perf = metriques_performance(source_map, cache)

    print("Calcul metriques qualite linguistique...")
    qualite = metriques_qualite(final_map)

    print("Calcul metriques diversification...")
    divers = metriques_diversification(source_map, final_map)

    print("Calcul metriques couverture...")
    couv = metriques_couverture(source_map, final_map, eval_p2)

    print("Selection echantillons comparatifs...")
    echant = echantillons_comparatifs(source_map, final_map, n_par_champ=5)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    version_info = (
        f"Genere le {ts} | "
        f"{couv['n_prenoms']:,} prenoms | "
        f"{couv['n_groupes']:,} groupes | "
        f"Modele : camembert2camembert_shared-finetuned-french-summarization"
    )

    print("Generation HTML...")
    html = generer_html(perf, qualite, divers, couv, echant, version_info)

    with open(OUTPUT_RAPPORT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Rapport genere : {OUTPUT_RAPPORT}")
    print(f"  Cache hit : {perf['taux_hit_pct']}%")
    print(f"  Temps economise : {perf['temps_economise_h']:.1f}h")
    for champ in CHAMPS:
        d = divers.get(champ, {})
        sc = (d.get("sim_cosine") or {}).get("moy")
        tr = d.get("taux_repli", "?")
        print(f"  {champ:15s} sim={sc:.3f if sc else '—'}  repli={tr}%")


if __name__ == "__main__":
    main()
