# Script PowerShell d'aide pour préparer l'environnement (Windows)
# Usage : Exécuter depuis la racine du projet
# .\scripts\setup_env.ps1

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Write-Host "Environnement configuré. Si l'installation de scikit-learn échoue, installez via conda ou utilisez Python 3.10/3.11."