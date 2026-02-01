"""
Script d'orchestration : exécute les étapes principales du pipeline dans l'ordre :
 1) Preprocessing
 2) Extraction de features
 3) Entraînement
 4) Évaluation

Ce script appelle les fonctions principales des scripts existants.
"""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(__file__))

def run(cmd):
    print('\n>>>', ' '.join(cmd))
    res = subprocess.run(cmd, shell=False)
    if res.returncode != 0:
        print('Erreur lors de l\'execution de :', cmd)
        sys.exit(res.returncode)


def main():
    # 1) Préprocessing
    run([sys.executable, os.path.join(ROOT, 'src', 'preprocessing_script.py')])

    # 2) Feature extraction
    run([sys.executable, os.path.join(ROOT, 'src', 'feature_extraction_script.py')])

    # 3) Train
    run([sys.executable, os.path.join(ROOT, 'src', 'train_model_script.py')])

    # 4) Evaluate
    run([sys.executable, os.path.join(ROOT, 'src', 'evaluate_script.py')])

    print('\nPipeline exécuté avec succès.')

if __name__ == '__main__':
    main()
