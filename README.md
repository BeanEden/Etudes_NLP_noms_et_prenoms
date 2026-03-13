# 🏷️ Etymia - NLP Name Processing

**Etymia** est un outil avancé de traitement et d'analyse de données onomastiques (noms et prénoms) utilisant des techniques de Traitement du Langage Naturel (NLP). Le projet permet de nettoyer, regrouper, résumer et enrichir les données de noms français avec des statistiques officielles de l'INSEE.

---

## Fonctionnalités

Le projet est divisé en deux pipelines principaux et une interface web :

### Pipeline Prénoms
1. **Préparation NLP** : Traitement initial des chaînes de caractères.
2. **Nettoyage & Formatage** : Normalisation des données sources.
3. **Regroupement** : Identification des variantes et regroupement par racines.
4. **Résumé** : Génération de statistiques synthétiques.
5. **Enrichissement INSEE** : Intégration des données démographiques historiques.

### Pipeline Noms de Famille
- Processus similaire incluant la préparation, le regroupement par proximité phonétique ou étymologique, le résumé et l'intégration des données INSEE.

### Interface Web (Flask)
- Une plateforme interactive pour explorer les résultats des pipelines et visualiser les statistiques.

---

## Installation

1. **Cloner le projet**
   ```bash
   git clone <url-du-repo>
   cd Projet
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sur Linux/Mac
   .venv\Scripts\activate     # Sur Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation

Le projet dispose d'un orchestrateur central pour simplifier l'exécution des différentes étapes :

```bash
python run_pipeline.py
```

Ce script interactif vous permet de :
- Lancer les pipelines complets ou par étapes.
- Démarrer le serveur Web Flask pour la visualisation.

---

## Structure du Projet

- `prenoms/` : Scripts et données relatifs aux prénoms.
- `noms/` : Scripts et données relatifs aux noms de famille.
- `flask/` : Application web pour l'interface utilisateur.
- `data/` : Dossier racine pour le stockage des données partagées.
- `run_pipeline.py` : Point d'entrée principal du projet.
- `devoir.pdf` : Consignes et spécifications du projet.

---

## Auteur
*Projet réalisé dans le cadre du Master 2 NLP - Sup de Vinci.*
