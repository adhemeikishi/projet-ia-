# Projet : Développement d'un modèle simple de classification d'images en Python

Ce dépôt contient un projet pédagogique visant à apprendre les bases de la classification d'images avec des modèles classiques (KNN, SVM, Régression Logistique). Toutes les étapes sont exécutées localement sans recours à des API externes ni à du deep learning.

## Objectif
Mettre en pratique : Python, NumPy, Pandas, Scikit-Learn, prétraitement d'images, extraction de caractéristiques, entraînement local, évaluation et automatisation des scripts.

## Organisation du dépôt
- data/
  - raw/ : images brutes organisées par classe (ex : `chat/`, `chien/`, `voiture/`)
  - processed/ : fichiers pickled (`preprocessed_data.pkl`, `features.pkl`)
- src/
  - dataset_downloader.py
  - preprocessing_script.py
  - feature_extraction_script.py
  - train_model_script.py
  - evaluate_script.py
  - predict_script.py
  - utils.py
- notebooks/ : notebook pédagogique pas-à-pas
- models/ : modèles sauvegardés
- results/ : graphiques et rapport texte
- scripts/
  - create_dummy_dataset.py : génère un petit dataset de test
  - run_pipeline.py : orchestration des étapes (préprocessing→features→train→evaluate)
  - test_pipeline.py : exécution de validation rapide
  - setup_env.ps1 : script PowerShell d'aide à l'installation (Windows)
- requirements.txt
- README.md (ce fichier)

## Pré-requis et installation
Recommandation : Python 3.10 ou 3.11 sur Windows pour éviter les problèmes d'installation de `scikit-learn` (les versions plus récentes peuvent nécessiter des build tools). Si vous utilisez `conda` / `mamba`, l'installation de `scikit-learn` est plus simple (roues précompilées).

1. Créez et activez un environnement virtuel :

Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

**Remarque** : si l'installation de `scikit-learn` échoue (par exemple sur Python 3.14 + Windows sans build tools), utilisez Python 3.10 / 3.11 ou installez via `conda`.

## Commandes principales
- Générer un dataset dummy (pour tests rapides) :

```bash
python scripts/create_dummy_dataset.py
```

- Exécuter tout le pipeline (préprocessing → features → train → evaluate) :

```bash
python scripts/run_pipeline.py
```

- Lancer un test de vérification rapide :

```bash
python scripts/test_pipeline.py
```

- Prédire une image :

```bash
python src/predict_script.py chemin/vers/image.jpg
```

## Démarche pédagogique (rapide)
1. Chargement des images (organisation par dossiers classes)
2. Prétraitement (redimensionnement, conversion RGB, normalisation)
3. Extraction de caractéristiques (histogrammes couleurs, statistiques de régions, option pixels bruts)
4. Standardisation des features
5. Entraînement de modèles classiques (KNN,SVM,Logistic Regression) et comparaison
6. Évaluation détaillée (matrice de confusion, precision/recall/F1, rapport)
7. Déploiement local via script de prédiction

## Justification technique
- Modèles choisis : KNN (simple et pédagogique), SVM (kernel pour non-linéarité), Régression Logistique (baseline probabiliste)
- Standardisation : importante pour KNN, SVM et Logistic Regression
- Métriques : accuracy pour vue globale + precision/recall/F1 et matrice de confusion pour comprendre les erreurs par classe

## Aide pour la soutenance orale
- Expliquez le pipeline (étapes, pourquoi standardiser, pourquoi choisir ces méthodes)
- Présentez des résultats (matrice de confusion et métriques par classe) pour montrer forces/faiblesses
- Mentionnez limitations : pas de deep learning, features manuelles, performance dépendante de qualité des images
