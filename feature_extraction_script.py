"""
Script 2 : Extraction de Caractéristiques (Features)

Ce script transforme les images en vecteurs de nombres exploitables
par les modèles de Machine Learning classiques.

Stratégie d'extraction :
1. Histogrammes de couleurs (distribution RGB)
2. Statistiques par région (moyenne, écart-type)
3. Pixels bruts (optionnel)

Compétences : NumPy, Pandas, Extraction de caractéristiques
"""

import numpy as np
import pandas as pd
import pickle
import os

INPUT_FILE = 'data/processed/preprocessed_data.pkl'
OUTPUT_FILE = 'data/processed/features.pkl'


def extraire_histogramme_couleurs(image, nb_bins=16):
    """
    Calcule l'histogramme des couleurs pour chaque canal RGB.
    
    L'histogramme compte combien de pixels ont chaque niveau de couleur.
    Cela capture la distribution des couleurs dans l'image.
    
    Args:
        image: array NumPy (hauteur, largeur, 3)
        nb_bins: nombre de bins pour l'histogramme
    
    Returns:
        features: vecteur 1D de taille (nb_bins * 3)
    """
    features = []
    
    # Pour chaque canal de couleur (R, G, B)
    for canal in range(3):
        # Calcule l'histogramme pour ce canal
        hist, _ = np.histogram(image[:, :, canal], bins=nb_bins, range=(0, 1))
        
        # Normalise l'histogramme (pour que la somme = 1)
        hist = hist / hist.sum()
        
        features.extend(hist)
    
    return np.array(features)


def extraire_statistiques_regions(image, grille=(4, 4)):
    """
    Divise l'image en grille et calcule des statistiques pour chaque région.
    
    Cette approche capture à la fois les informations de couleur ET
    la structure spatiale de l'image.
    
    Args:
        image: array NumPy (hauteur, largeur, 3)
        grille: tuple (nb_lignes, nb_colonnes)
    
    Returns:
        features: vecteur 1D contenant moyenne et écart-type de chaque région
    """
    hauteur, largeur, canaux = image.shape
    nb_lignes, nb_colonnes = grille
    
    # Calcule la taille de chaque région
    taille_h = hauteur // nb_lignes
    taille_w = largeur // nb_colonnes
    
    features = []
    
    # Pour chaque région de la grille
    for i in range(nb_lignes):
        for j in range(nb_colonnes):
            # Découpe la région
            region = image[i*taille_h:(i+1)*taille_h, 
                          j*taille_w:(j+1)*taille_w, :]
            
            # Calcule des statistiques pour chaque canal
            for canal in range(canaux):
                region_canal = region[:, :, canal]
                
                # Moyenne : valeur moyenne des pixels
                moyenne = np.mean(region_canal)
                
                # Ecart-type : mesure de la variation des pixels
                ecart_type = np.std(region_canal)
                
                features.extend([moyenne, ecart_type])
    
    return np.array(features)


def extraire_pixels_bruts(image):
    """
    Aplatit l'image en un vecteur 1D.
    
    Simple mais peu performant car ne capture pas les patterns.
    Utilisé ici pour comparaison.
    
    Args:
        image: array NumPy (hauteur, largeur, 3)
    
    Returns:
        features: vecteur 1D de tous les pixels
    """
    return image.flatten()


def extraire_features_completes(image, inclure_pixels=False):
    """
    Combine plusieurs types de features pour une image.
    
    Args:
        image: array NumPy (hauteur, largeur, 3)
        inclure_pixels: si True, ajoute aussi les pixels bruts
    
    Returns:
        features: vecteur 1D combinant toutes les features
    """
    features = []
    
    # 1. Histogrammes de couleurs (16 bins par canal = 48 features)
    hist_features = extraire_histogramme_couleurs(image, nb_bins=16)
    features.extend(hist_features)
    
    # 2. Statistiques par région (4x4 grille, 2 stats par canal = 96 features)
    stats_features = extraire_statistiques_regions(image, grille=(4, 4))
    features.extend(stats_features)
    
    # 3. Pixels bruts (optionnel, 32*32*3 = 3072 features)
    if inclure_pixels:
        pixels_features = extraire_pixels_bruts(image)
        features.extend(pixels_features)
    
    return np.array(features)


def extraire_features_dataset(X, inclure_pixels=False):
    """
    Extrait les features pour toutes les images du dataset.
    
    Args:
        X: array NumPy (nb_images, hauteur, largeur, 3)
        inclure_pixels: si True, inclut aussi les pixels bruts
    
    Returns:
        X_features: array NumPy (nb_images, nb_features)
    """
    print(f"Extraction des features pour {len(X)} images...")
    
    features_list = []
    
    # Pour chaque image
    for i, image in enumerate(X):
        # Extrait les features
        features = extraire_features_completes(image, inclure_pixels)
        features_list.append(features)
        
        # Affiche la progression tous les 100 images
        if (i + 1) % 100 == 0:
            print(f"  -> {i + 1}/{len(X)} images traitees")
    
    # Convertit en array NumPy
    X_features = np.array(features_list)
    
    print(f"Extraction terminee : {X_features.shape[1]} features par image")
    
    return X_features


def creer_dataframe_features(X_features, y, classes_noms):
    """
    Crée un DataFrame Pandas avec les features et les labels.
    
    Cela facilite l'analyse et la manipulation des données.
    
    Args:
        X_features: array NumPy des features
        y: array NumPy des labels
        classes_noms: liste des noms de classes
    
    Returns:
        df: DataFrame Pandas
    """
    # Crée les noms de colonnes
    nb_features = X_features.shape[1]
    noms_colonnes = [f'feature_{i}' for i in range(nb_features)]
    
    # Crée le DataFrame avec les features
    df = pd.DataFrame(X_features, columns=noms_colonnes)
    
    # Ajoute la colonne des labels numériques
    df['label'] = y
    
    # Ajoute la colonne des noms de classes
    df['classe'] = df['label'].apply(lambda x: classes_noms[x])
    
    return df


def main():
    """
    Fonction principale pour l'extraction de features.
    """
    print("=" * 50)
    print("ÉTAPE 2 : EXTRACTION DE CARACTÉRISTIQUES")
    print("=" * 50)
    
    # Charge les données prétraitées
    print(f"\nChargement des donnees depuis {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERREUR : Le fichier {INPUT_FILE} n'existe pas.")
        print("Veuillez d'abord executer preprocessing_script.py")
        return
    
    with open(INPUT_FILE, 'rb') as f:
        donnees = pickle.load(f)
    
    X = donnees['X']
    y = donnees['y']
    classes_noms = donnees['classes_noms']
    
    print(f"Donnees chargees : {X.shape[0]} images")
    
    # Extrait les features
    print("\n" + "=" * 50)
    print("EXTRACTION DES FEATURES")
    print("=" * 50)
    
    # Choix : inclure ou non les pixels bruts
    # Pour ce projet, on ne les inclut pas (trop de features)
    X_features = extraire_features_dataset(X, inclure_pixels=False)
    
    # Crée un DataFrame Pandas
    print("\nCreation du DataFrame Pandas...")
    df = creer_dataframe_features(X_features, y, classes_noms)
    
    # Affiche les statistiques
    print("\n" + "=" * 50)
    print("STATISTIQUES DES FEATURES")
    print("=" * 50)
    print(f"Nombre d'images : {len(df)}")
    print(f"Nombre de features : {X_features.shape[1]}")
    print(f"\nPremiers apercu du DataFrame :")
    print(df.head())
    
    print(f"\nStatistiques descriptives :")
    print(df[['feature_0', 'feature_1', 'feature_2']].describe())
    
    # Sauvegarde les features
    print("\n" + "=" * 50)
    print("SAUVEGARDE DES FEATURES")
    print("=" * 50)
    
    donnees_features = {
        'X_features': X_features,
        'y': y,
        'classes_noms': classes_noms,
        'df': df
    }
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(donnees_features, f)
    
    print(f"Features sauvegardees dans : {OUTPUT_FILE}")
    
    print("\n" + "=" * 50)
    print("EXTRACTION TERMINÉE AVEC SUCCÈS")
    print("=" * 50)


if __name__ == "__main__":
    main()
