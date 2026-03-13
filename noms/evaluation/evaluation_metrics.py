import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── Compatibilité chemin : import depuis n'importe quel CWD ──────────────────
# Permet d'appeler `import evaluation_metrics` depuis un notebook ouvert
# dans noms/ ou depuis la racine du projet.

def load_evaluation_data(filepath):
    """Charge les données d'évaluation (paires de noms)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_clustering_metrics(eval_data):
    """
    Calcule Précision / Rappel / F1 et la matrice de confusion.
    Attend une liste de dicts avec les clés 'attendu', 'groupe_a', 'groupe_b'.
    """
    y_true, y_pred = [], []

    for item in eval_data:
        true_val = 1 if item.get('attendu') == 'lie' else 0
        pred_val = 1 if item.get('groupe_a') == item.get('groupe_b') else 0
        y_true.append(true_val)
        y_pred.append(pred_val)

    if not y_true:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'confusion_matrix': [[0,0],[0,0]]}

    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        # On conserve en liste pour la sérialisation JSON, plot_confusion_matrix sait gérer
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics


def plot_confusion_matrix(cm, title='Matrice de Confusion'):
    """
    Affiche la matrice de confusion.
    Accepte indifféremment un np.ndarray ou une liste Python.
    """
    cm_arr = np.array(cm)   # ← Conversion list → ndarray si nécessaire
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt='d',
        cmap='Oranges',
        xticklabels=['Non Lié', 'Lié'],
        yticklabels=['Non Lié', 'Lié'],
        linewidths=0.5,
        linecolor='white',
    )
    plt.xlabel('Prédiction (Même Groupe)', fontsize=11)
    plt.ylabel('Réalité (Attendu)', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ── Utilitaire interne ──────────────────────────────────────────────────────

def _iter_groups(groups_data):
    """
    Normalise l'itération sur les groupes, quelle que soit la structure :
      - dict  { id: { 'noms': [...] | 'membres': [...] } }
      - list  [ { 'id_groupe_total': ..., 'noms': [...] | 'membres': [...] } ]
    Retourne un itérateur de (gid, liste_de_membres).
    """
    if isinstance(groups_data, dict):
        items = ((gid, g) for gid, g in groups_data.items())
    else:
        items = ((g.get('id_groupe_total', i), g) for i, g in enumerate(groups_data))

    for gid, g in items:
        membres = g.get('noms') or g.get('membres') or g.get('prenoms') or []
        yield gid, membres


def analyze_groups(groups_file):
    """Analyse la distribution des tailles de groupes de noms."""
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = json.load(f)

    sizes = [len(m) for _, m in _iter_groups(groups)]
    n_total = len(sizes)
    n_singletons = sum(1 for s in sizes if s == 1)

    print(f"Groupes total       : {n_total}")
    print(f"Dont singletons     : {n_singletons} ({n_singletons/n_total:.1%})")
    print(f"Taille moyenne      : {np.mean(sizes):.2f}")
    print(f"Maximum             : {max(sizes)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution brute
    axes[0].set_title("Distribution de la taille des groupes (brut)", fontweight='bold')
    sns.histplot(sizes, bins=50, ax=axes[0], color='#B8973A')
    axes[0].set_yscale('log')
    axes[0].set_xlabel("Taille du groupe")
    axes[0].set_ylabel("Nombre de groupes (log)")

    # Zoom sur groupes > 1
    multi = [s for s in sizes if s > 1]
    if multi:
        axes[1].set_title("Groupes avec ≥ 2 membres", fontweight='bold')
        sns.histplot(multi, bins=min(30, len(set(multi))), ax=axes[1], color='#1A2B4A')
        axes[1].set_xlabel("Taille du groupe")
        axes[1].set_ylabel("Nombre de groupes")

    plt.tight_layout()
    plt.show()


def compute_language_consistency(groups_file, noms_data_file):
    """
    Calcule la cohérence d'origine au sein des groupes.
    Un groupe est cohérent si ≥75% des membres partagent la même origine.
    """
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    with open(noms_data_file, 'r', encoding='utf-8') as f:
        noms_raw = json.load(f)

    # Support dict ou list pour noms_raw
    if isinstance(noms_raw, list):
        noms_to_origin = {
            d.get('nom', ''): d.get('geo', d.get('langue', d.get('origine', '')))
            for d in noms_raw
        }
    else:
        noms_to_origin = {k: v.get('geo', v.get('langue', '')) for k, v in noms_raw.items()}

    consistent_groups = 0
    total_multi_groups = 0

    for _, membres in _iter_groups(groups):
        if len(membres) <= 1:
            continue
        total_multi_groups += 1

        langs = [noms_to_origin.get(m, '') for m in membres if noms_to_origin.get(m, '')]
        if not langs:
            consistent_groups += 1
            continue

        counts = Counter(langs)
        _, count = counts.most_common(1)[0]
        if count / len(langs) >= 0.75:
            consistent_groups += 1

    ratio = consistent_groups / total_multi_groups if total_multi_groups > 0 else 1.0
    return ratio, total_multi_groups


def compute_confidence_scores(groups_file, data_file):
    """
    Score de cohésion interne des groupes via similarité cosinus TF-IDF
    sur les textes d'origine des noms.
    """
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        nom_to_text = {
            d.get('nom', ''): d.get('origine_brute', d.get('texte', d.get('resume', '')))
            for d in data
        }
    else:
        nom_to_text = {k: v.get('origine_brute', '') for k, v in data.items()}

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    all_scores = []
    for _, membres in _iter_groups(groups):
        if len(membres) <= 1:
            continue
        textes = [nom_to_text.get(m, '') or '__vide__' for m in membres]
        try:
            vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            tfidf = vec.fit_transform(textes)
            sims = cosine_similarity(tfidf)
            for i in range(len(membres)):
                score = (np.sum(sims[i]) - 1.0) / max(len(membres) - 1, 1)
                all_scores.append(score)
        except Exception:
            continue

    return np.mean(all_scores) if all_scores else 0.0


def plot_score_distribution(groups_file, data_file, title="Distribution des scores de confiance"):
    """
    Affiche l'histogramme de la distribution des scores de cohésion interne.
    """
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        nom_to_text = {
            d.get('nom', ''): d.get('origine_brute', d.get('texte', d.get('resume', '')))
            for d in data
        }
    else:
        nom_to_text = {k: v.get('origine_brute', '') for k, v in data.items()}

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    all_scores = []
    for _, membres in _iter_groups(groups):
        if len(membres) <= 1:
            continue
        textes = [nom_to_text.get(m, '') or '__vide__' for m in membres]
        try:
            vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            tfidf = vec.fit_transform(textes)
            sims = cosine_similarity(tfidf)
            for i in range(len(membres)):
                score = (np.sum(sims[i]) - 1.0) / max(len(membres) - 1, 1)
                all_scores.append(score)
        except Exception:
            continue

    plt.figure(figsize=(10, 5))
    sns.histplot(all_scores, bins=40, color='#B8973A', kde=True)
    plt.axvline(np.mean(all_scores), color='#1A2B4A', linestyle='--', label=f"Moyenne : {np.mean(all_scores):.2f}")
    plt.xlabel("Score de confiance interne")
    plt.ylabel("Nombre de membres")
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return all_scores
