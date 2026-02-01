"""
Script de test rapide : vérifie que les fichiers attendus sont produits par le pipeline
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

FILES_EXPECTED = [
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'preprocessed_data.pkl'),
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'features.pkl'),
    os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl'),
    os.path.join(PROJECT_ROOT, 'results', 'rapport_evaluation.txt')
]

# Exécute le pipeline (silencieux)
res = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'scripts', 'run_pipeline.py')])
if res.returncode != 0:
    print('Erreur lors de l\'execution du pipeline')
    sys.exit(res.returncode)

# Vérifie la présence des fichiers
ok = True
for f in FILES_EXPECTED:
    if not os.path.exists(f):
        print('Fichier attendu manquant :', f)
        ok = False

if ok:
    print('Test pipeline : SUCCESS — tous les fichiers attendus sont présents')
else:
    print('Test pipeline : ECHEC — voir les fichiers manquants')
    sys.exit(2)
