"""
Script pour exécuter tous les notebooks Jupyter dans l'ordre
"""
import subprocess
import sys
import os
from pathlib import Path

def run_notebook(notebook_path):
    """Exécute un notebook Jupyter"""
    print(f"\n{'='*60}")
    print(f"Exécution de {notebook_path.name}")
    print(f"{'='*60}")
    
    try:
        # Utiliser jupyter nbconvert pour exécuter le notebook
        result = subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", 
             "--inplace", str(notebook_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[OK] {notebook_path.name} exécuté avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERREUR] Erreur lors de l'exécution de {notebook_path.name}")
        if e.stdout:
            print(f"  Sortie : {e.stdout[-500:]}")  # Derniers 500 caractères
        if e.stderr:
            print(f"  Erreur : {e.stderr[-500:]}")
        return False
    except FileNotFoundError:
        print(f"[ERREUR] Jupyter n'est pas installé ou n'est pas dans le PATH")
        print(f"  Installez avec : pip install jupyter")
        return False

def main():
    """Exécute tous les notebooks dans l'ordre"""
    notebooks_dir = Path("notebooks")
    
    # Ordre d'exécution des notebooks
    notebooks = [
        "03_ml_data_generation.ipynb",
        "04_ml_training.ipynb",
        "05_ml_integration.ipynb"
    ]
    
    print("="*60)
    print("EXÉCUTION DE TOUS LES NOTEBOOKS JUPYTER")
    print("="*60)
    print("\nNote : Assurez-vous d'avoir d'abord exécuté :")
    print("  - src/simulate_warehouse.py (pour générer les données de base)")
    print("\nOrdre d'exécution :")
    for i, nb in enumerate(notebooks, 1):
        print(f"  {i}. {nb}")
    
    all_success = True
    for nb_name in notebooks:
        nb_path = notebooks_dir / nb_name
        if not nb_path.exists():
            print(f"\n[ERREUR] Notebook introuvable : {nb_path}")
            all_success = False
            continue
        
        success = run_notebook(nb_path)
        if not success:
            all_success = False
            print(f"\nArrêt de l'exécution à cause d'une erreur dans {nb_name}")
            print("Vérifiez les messages d'erreur ci-dessus.")
            break
    
    if all_success:
        print(f"\n{'='*60}")
        print("[OK] TOUS LES NOTEBOOKS ONT ÉTÉ EXÉCUTÉS AVEC SUCCÈS")
        print(f"{'='*60}")
        print("\nRésultats disponibles dans :")
        print("  - data/raw/ : Datasets générés")
        print("  - data/processed/ : Modèles et résultats")
    else:
        print(f"\n{'='*60}")
        print("[ERREUR] ERREURS DÉTECTÉES")
        print(f"{'='*60}")
        print("\nVérifiez les messages d'erreur ci-dessus.")
        print("Assurez-vous d'avoir exécuté les étapes préalables :")
        print("  1. src/simulate_warehouse.py")
        print("  2. Les notebooks dans l'ordre")
        sys.exit(1)

if __name__ == "__main__":
    main()
