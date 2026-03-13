
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_evaluation_data(filepath):
    """Charge les données d'évaluation (paires de prénoms)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_clustering_metrics(eval_data):
    """
    Calcule les métriques de performance basées sur les étiquettes 'attendu'
    vs le fait que les prénoms soient effectivement dans le même groupe.
    """
    y_true = []
    y_pred = []
    
    for item in eval_data:
        # On définit 'lie' comme 1 et 'non_lie' comme 0
        true_val = 1 if item['attendu'] == 'lie' else 0
        
        # Le modèle a 'prédit' un lien si les deux prénoms sont dans le même groupe
        pred_val = 1 if item['groupe_a'] == item['groupe_b'] else 0
        
        y_true.append(true_val)
        y_pred.append(pred_val)
        
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def plot_confusion_matrix(cm, title='Matrice de Confusion - Clustering'):
    """Affiche la matrice de confusion."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non Lié', 'Lié'],
                yticklabels=['Non Lié', 'Lié'])
    plt.xlabel('Prédiction (Même Groupe)')
    plt.ylabel('Réalité (Attendu)')
    plt.title(title)
    plt.show()

def analyze_groups(groups_file):
    """Analyse la distribution des groupes."""
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    
    sizes = [len(g['membres']) for g in groups.values()]
    
    print(f"Nombre total de groupes : {len(groups)}")
    print(f"Taille moyenne : {np.mean(sizes):.2f}")
    print(f"Médiane : {np.median(sizes)}")
    print(f"Maximum : {max(sizes)}")
    
    plt.figure(figsize=(10, 5))
    sns.histplot(sizes, bins=50, kde=True)
    plt.title("Distribution de la taille des groupes")
    plt.xlabel("Nombre de membres")
    plt.ylabel("Nombre de groupes")
    plt.yscale('log') # Souvent utile pour les noms/prénoms
    plt.show()

def compute_language_consistency(groups_file, prenoms_data_file):
    """
    Calcule le pourcentage de groupes ayant une origine linguistique unique (ou dominante).
    """
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    with open(prenoms_data_file, 'r', encoding='utf-8') as f:
        prenoms_data = {d['prenom']: d.get('langue', '') for d in json.load(f)}
    
    consistent_groups = 0
    total_multi_groups = 0
    
    for gid, g in groups.items():
        if len(g['membres']) <= 1:
            continue
            
        total_multi_groups += 1
        langs = [prenoms_data.get(m, '') for m in g['membres'] if prenoms_data.get(m, '')]
        if not langs:
            consistent_groups += 1
            continue
            
        counts = Counter(langs)
        most_common_lang, count = counts.most_common(1)[0]
        # On considère cohérent si > 75% partagent la même langue
        if count / len(langs) >= 0.75:
            consistent_groups += 1
            
    ratio = consistent_groups / total_multi_groups if total_multi_groups > 0 else 1.0
    return ratio, total_multi_groups

def compute_confidence_scores(eval_file):
    """
    Extrait le score de confiance moyen (sim_centroid) depuis le fichier d'évaluation.
    """
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # On peut aussi parcourir les outliers pour voir la distribution des scores bas
    outliers_data = eval_data.get('outliers', [])
    all_sims = []
    
    # Note: sim_centroid n'est disponible que pour les groupes d'evaluation ou chargés spécifiquement.
    # Ici on va se baser sur les paires liées de l'échantillon d'évaluation comme proxy 
    # de la confiance du modèle sur ses propres fusions.
    
    paires = eval_data.get('paires_evaluation', [])
    for p in paires:
        if p['attendu'] == 'lie' and p['groupe_a'] == p['groupe_b']:
            # Ces paires sont des fusions validées par le modèle
            # On n'a pas le score exact ici mais on peut dire que c'est >= SEUIL_SCORE_FINAL
            pass

    # Alternative : le fichier eval_prenoms.json contient les outliers avec sim_centroid.
    # On va faire la moyenne des scores de tous les membres mentionnés dans la section outliers
    for o in outliers_data:
        for p in o['outliers']:
            all_sims.append(p['sim_centroid'])
            
    return np.mean(all_sims) if all_sims else 0.88 # 0.88 est le SEUIL_CENTROID
