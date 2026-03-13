import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_evaluation_data(filepath):
    """Charge les données d'évaluation (paires de noms)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_clustering_metrics(eval_data):
    """
    Calcule les métriques de performance.
    """
    y_true = []
    y_pred = []
    
    for item in eval_data:
        true_val = 1 if item['attendu'] == 'lie' else 0
        pred_val = 1 if item['groupe_a'] == item['groupe_b'] else 0
        
        y_true.append(true_val)
        y_pred.append(pred_val)
        
    metrics = {
        'precision': precision_score(y_true, y_pred) if y_true else 0,
        'recall': recall_score(y_true, y_pred) if y_true else 0,
        'f1': f1_score(y_true, y_pred) if y_true else 0,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist() if y_true else [[0,0],[0,0]]
    }
    return metrics

def analyze_groups(groups_file):
    """Analyse la distribution des groupes de noms."""
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    
    sizes = [len(g['membres']) for g in groups.values()]
    
    print(f"Nombre total de groupes : {len(groups)}")
    print(f"Taille moyenne : {np.mean(sizes):.2f}")
    print(f"Maximum : {max(sizes)}")
    
    plt.figure(figsize=(10, 5))
    sns.histplot(sizes, bins=50)
    plt.title("Distribution de la taille des groupes (Noms)")
    plt.yscale('log')
    plt.show()

def compute_language_consistency(groups_file, noms_data_file):
    """
    Calcule la cohérence linguistique (ou géographique pour les noms).
    """
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    with open(noms_data_file, 'r', encoding='utf-8') as f:
        noms_data = {d['nom']: d.get('origine', '') for d in json.load(f)}
    
    consistent_groups = 0
    total_multi_groups = 0
    
    for gid, g in groups.items():
        if len(g['membres']) <= 1:
            continue
            
        total_multi_groups += 1
        langs = [noms_data.get(m, '') for m in g['membres'] if noms_data.get(m, '')]
        if not langs:
            consistent_groups += 1
            continue
            
        counts = Counter(langs)
        _, count = counts.most_common(1)[0]
        if count / len(langs) >= 0.75:
            consistent_groups += 1
            
    ratio = consistent_groups / total_multi_groups if total_multi_groups > 0 else 1.0
    return ratio, total_multi_groups

def plot_confusion_matrix(cm, title='Matrice de Confusion'):
    """Affiche la matrice de confusion."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Non Lié', 'Lié'],
                yticklabels=['Non Lié', 'Lié'])
    plt.xlabel('Prédiction (Même Groupe)')
    plt.ylabel('Réalité (Attendu)')
    plt.title(title)
    plt.show()

def compute_confidence_scores(groups_file, data_file):
    """
    Calcule un score de confiance moyen pour chaque nom dans son groupe.
    Basé sur la similarité TF-IDF moyenne avec les autres membres.
    """
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    noms_to_text = {d['nom']: d.get('origine_brute', '') for d in data}
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    all_scores = []
    
    for gid, g in groups.items():
        membres = g['noms']
        if len(membres) <= 1:
            continue
            
        textes = [noms_to_text.get(m, '') or '__vide__' for m in membres]
        vec = TfidfVectorizer(analyzer='char', ngram_range=(2,3)) # Caractères pour les noms
        try:
            tfidf = vec.fit_transform(textes)
            sims = cosine_similarity(tfidf)
            # Moyenne des similarités pour chaque membre (excluant soi-même)
            for i in range(len(membres)):
                # sum of all sim in row i minus 1.0 (self sim) / (N-1)
                score = (np.sum(sims[i]) - 1.0) / (len(membres) - 1)
                all_scores.append(score)
        except:
            continue
            
    return np.mean(all_scores) if all_scores else 0.0
