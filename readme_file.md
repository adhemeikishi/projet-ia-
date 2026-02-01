# Projet : Classification d'Images avec Machine Learning Classique

Projet académique développé en Python pour la classification d'images utilisant des algorithmes de Machine Learning classiques (sans deep learning).

## Description du Projet

Ce projet met en œuvre un pipeline complet de classification d'images en utilisant :
- Extraction manuelle de caractéristiques (features engineering)
- Modèles de Machine Learning classiques (KNN, SVM, Régression Logistique)
- Entraînement et évaluation locaux (sans API externe)

## Technologies Utilisées

- **Python 3.8+**
- **NumPy** : calculs numériques et manipulation de tableaux
- **Pandas** : analyse et structuration des données
- **Scikit-Learn** : modèles de Machine Learning et métriques
- **Pillow** : traitement d'images
- **Matplotlib / Seaborn** : visualisation des résultats

## Structure du Projet

```
classification-images/
│
├── data/
│   ├── raw/                          # Images brutes organisées par classe
│   │   ├── classe_1/
│   │   ├── classe_2/
│   │   └── classe_3/
│   └── processed/                    # Données prétraitées
│       ├── preprocessed_data.pkl
│       └── features.pkl
│
├── models/                           # Modèles entraînés
│   ├── best_model.pkl
│   ├── knn_k5.pkl
│   ├── svm_rbf.pkl
│   └── regression_logistique.pkl
│
├── results/                          # Résultats d'évaluation
│   ├── matrice_confusion.png
│   ├── metriques_par_classe.png
│   └── rapport_evaluation.txt
│
├── src/
│   ├── dataset_downloader.py       # Téléchargement du dataset (optionnel)
│   ├── preprocessing_script.py          # Prétraitement des images
│   ├── feature_extraction_script.py     # Extraction de caractéristiques
│   ├── train_model_script.py            # Entraînement des modèles
│   ├── evaluate_script.py               # Évaluation détaillée
│   └── predict_script.py                # Prédiction sur nouvelles images
│
├── requirements.txt                  # Dépendances Python
└── README.md                         # Ce fichier
```

## Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd classification-images
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Activer l'environnement
# Windows :
venv\Scripts\activate
# Linux/Mac :
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Remarque** : si l'installation de `scikit-learn` échoue (par ex. avec Python 3.14 sur Windows, manque de build tools), utilisez une version de Python supportée (3.10 ou 3.11) ou installez via `conda`/`mamba` pour obtenir des roues précompilées.


## Utilisation

### Étape 0 : Préparation du Dataset (OPTIONNEL)

Deux options possibles :

#### Option A : Télécharger CIFAR-10 automatiquement

```bash
python dataset_downloader.py
```

Ce script télécharge automatiquement un sous-ensemble du dataset CIFAR-10 et l'organise dans la structure attendue.

#### Option B : Utiliser votre propre dataset

Créez la structure suivante manuellement :

```
data/raw/
    chat/
        image1.jpg
        image2.jpg
        ...
    chien/
        image1.jpg
        image2.jpg
        ...
    voiture/
        image1.jpg
        image2.jpg
        ...
```

**Recommandations** :
- Minimum 100 images par classe
- Formats acceptés : JPG, PNG, BMP
- Les images seront automatiquement redimensionnées à 32x32 pixels

### Étape 1 : Prétraitement des Images

```bash
python preprocessing_script.py
```

**Ce script effectue** :
- Chargement des images depuis `data/raw/`
- Redimensionnement à 32x32 pixels
- Normalisation (valeurs entre 0 et 1)
- Sauvegarde dans `data/processed/preprocessed_data.pkl`

**Résultat** : Données prêtes pour l'extraction de caractéristiques

### Étape 2 : Extraction de Caractéristiques

```bash
python feature_extraction_script.py
```

**Ce script extrait** :
- **Histogrammes de couleurs** : distribution RGB (48 features)
- **Statistiques par région** : moyenne et écart-type sur grille 4x4 (96 features)
- Total : 144 features par image

**Résultat** : DataFrame Pandas sauvegardé dans `data/processed/features.pkl`

### Étape 3 : Entraînement des Modèles

```bash
python train_model_script.py
```

**Ce script** :
- Sépare les données en train (80%) et test (20%)
- Standardise les features
- Entraîne 3 modèles :
  - **K-Nearest Neighbors (k=5)**
  - **Support Vector Machine (kernel RBF)**
  - **Régression Logistique**
- Compare leurs performances
- Sauvegarde le meilleur modèle dans `models/best_model.pkl`

**Résultat** : Modèles entraînés et prêts pour l'évaluation

### Étape 4 : Évaluation Détaillée

```bash
python evaluate_script.py
```

**Ce script génère** :
- **Matrice de confusion** : visualise les confusions entre classes
- **Rapport de classification** : précision, rappel, F1-score par classe
- **Analyse des erreurs** : identifie les confusions les plus fréquentes
- **Visualisations** : graphiques des métriques

**Résultat** : Rapports et graphiques dans `results/`

### Étape 5 : Prédiction sur Nouvelles Images

#### Prédire une seule image

```bash
python predict_script.py chemin/vers/image.jpg
```

#### Prédire toutes les images d'un dossier

```bash
python predict_script.py chemin/vers/dossier/
```

#### Mode interactif

```bash
python predict_script.py
```

Le mode interactif vous permettra de saisir plusieurs chemins successivement.

## Pipeline Machine Learning

```
Images brutes (JPG/PNG)
    ↓
[1] Prétraitement
    - Redimensionnement 32x32
    - Conversion RGB
    - Normalisation (0-1)
    ↓
[2] Extraction de Features
    - Histogrammes couleurs
    - Statistiques spatiales
    ↓
Données tabulaires (144 features/image)
    ↓
[3] Split Train/Test (80/20)
    ↓
[3] Standardisation (μ=0, σ=1)
    ↓
[3] Entraînement
    - KNN, SVM, LogReg
    ↓
[4] Évaluation
    - Accuracy, Confusion Matrix
    - Précision, Rappel, F1
    ↓
[5] Prédiction
    - Nouvelles images
```

## Métriques d'Évaluation

### Accuracy
Pourcentage de prédictions correctes sur l'ensemble total.

```
Accuracy = (Vrais Positifs + Vrais Négatifs) / Total
```

### Précision
Parmi les prédictions positives, combien sont correctes.

```
Précision = Vrais Positifs / (Vrais Positifs + Faux Positifs)
```

### Rappel (Recall)
Parmi les vrais positifs, combien ont été détectés.

```
Rappel = Vrais Positifs / (Vrais Positifs + Faux Négatifs)
```

### F1-Score
Moyenne harmonique de la précision et du rappel.

```
F1 = 2 × (Précision × Rappel) / (Précision + Rappel)
```

## Choix des Modèles

### K-Nearest Neighbors (KNN)
- **Principe** : Classe une image selon ses k plus proches voisins
- **Avantages** : Simple, pas de phase d'entraînement
- **Inconvénients** : Lent en prédiction, sensible à l'échelle
- **Cas d'usage** : Baseline simple, bon pour comprendre

### Support Vector Machine (SVM)
- **Principe** : Trouve l'hyperplan séparant au mieux les classes
- **Avantages** : Performant, gère bien la complexité
- **Inconvénients** : Plus long à entraîner
- **Cas d'usage** : Meilleur compromis performance/complexité

### Régression Logistique
- **Principe** : Modèle linéaire probabiliste
- **Avantages** : Rapide, interprétable
- **Inconvénients** : Moins performant sur images complexes
- **Cas d'usage** : Baseline rapide

## Extraction de Caractéristiques Expliquée

### Pourquoi extraire des features ?

Les modèles classiques ne peuvent pas traiter directement les images. Ils nécessitent des vecteurs de nombres. L'extraction de features transforme une image en vecteur numérique tout en conservant l'information importante.

### Histogrammes de couleurs

Capture la distribution des couleurs dans l'image :
- 16 bins par canal RGB
- Indépendant de la position spatiale
- Utile pour distinguer des objets par couleur

### Statistiques par région

Capture à la fois couleur ET structure spatiale :
- Divise l'image en grille 4×4 (16 régions)
- Calcule moyenne et écart-type par région
- Préserve l'information de position

## Performances Attendues

Avec le dataset CIFAR-10 (3 classes) :
- **KNN** : 50-60% accuracy
- **SVM** : 60-70% accuracy
- **Régression Logistique** : 45-55% accuracy

Ces performances sont normales pour des modèles classiques sans deep learning. Pour comparaison, un CNN moderne atteint 90%+ sur CIFAR-10.

## Troubleshooting

### Erreur : "No module named 'sklearn'"

```bash
pip install scikit-learn
```

### Erreur : "data/raw/ n'existe pas"

Créez la structure de dossiers ou exécutez `dataset_downloader.py`. 

### Images mal classifiées

C'est normal avec des modèles classiques. Pour améliorer :
- Augmenter le nombre d'images d'entraînement
- Tester d'autres hyperparamètres (k pour KNN, C pour SVM)
- Ajouter plus de features (textures, contours)

## Améliorations Possibles

1. **Plus de features** : textures (GLCM), contours (Sobel), HOG
2. **Réduction de dimensionnalité** : PCA pour réduire les 144 features
3. **Validation croisée** : pour des résultats plus robustes
4. **Grid Search** : optimisation automatique des hyperparamètres
5. **Data Augmentation** : rotation, flip, zoom pour plus de données

## Auteur

Projet académique - Octobre 2025

## Licence

Ce projet est à usage éducatif uniquement.
