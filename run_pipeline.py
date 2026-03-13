#!/usr/bin/env python3
import os
import sys
import subprocess

# Define the root path relative to the script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINES = {
    "noms": [
        "1_prepare_noms.py",
        "2_regroupement_noms.py",
        "3_summarize_noms.py",
        "4_integrate_insee.py"
    ],
    "prenoms": [
        "0_prepare_nlp.py",
        "1_prepare_prenoms.py",
        "2_regroupement_prenoms.py",
        "3_summarize_prenoms.py",
        "4_enrichir_insee.py"
    ]
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print("\n" + "="*50)
    print(f"  {title}".center(50))
    print("="*50 + "\n")

def run_script(folder, script_name):
    cwd = os.path.join(ROOT_DIR, folder)
    print(f"\n Lancement de : {script_name}...")
    try:
        subprocess.run([sys.executable, script_name], cwd=cwd, check=True)
        print(f"\n Terminé avec succès : {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n Erreur d'exécution de {script_name} (Code {e.returncode})")
    input("\nAppuyez sur Entrée pour continuer...")

def run_all(folder):
    for script in PIPELINES[folder]:
        cwd = os.path.join(ROOT_DIR, folder)
        print(f"\n Lancement de : {script}...")
        try:
            subprocess.run([sys.executable, script], cwd=cwd, check=True)
            print(f"Succès : {script}")
        except subprocess.CalledProcessError as e:
            print(f"Erreur critique sur {script} (Code {e.returncode}). Arrêt du pipeline.")
            break
    input("\nAppuyez sur Entrée pour continuer...")

def menu_pipeline(folder):
    while True:
        clear_screen()
        print_header(f"Pipeline : {folder.upper()}")
        scripts = PIPELINES[folder]
        for i, s in enumerate(scripts, 1):
            print(f"  {i}. Exécuter {s}")
        print(f"  {len(scripts)+1}. Exécuter TOUT le pipeline {folder.upper()} (Séquentiel)")
        print("\n  0. Retour au menu principal")
        
        choice = input("\nVotre choix : ").strip()
        if choice == "0":
            break
        elif choice.isdigit():
            c = int(choice)
            if 1 <= c <= len(scripts):
                run_script(folder, scripts[c-1])
            elif c == len(scripts) + 1:
                run_all(folder)
        
def menu_principal():
    while True:
        clear_screen()
        print_header("Etymia - Gestionnaire de Pipelines NLP")
        print("  1. Pipeline NOMS DE FAMILLE")
        print("  2. Pipeline PRÉNOMS")
        print("  3. Lancer le serveur Web Flask")
        print("\n  0. Quitter")
        
        choice = input("\nVotre choix : ").strip()
        if choice == "0":
            print("Au revoir !\n")
            break
        elif choice == "1":
            menu_pipeline("noms")
        elif choice == "2":
            menu_pipeline("prenoms")
        elif choice == "3":
            run_script("flask", "app.py")

if __name__ == "__main__":
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\nArrêt manuel. Au revoir!")
        sys.exit(0)
