"""
Script 5 : Prédiction sur Nouvelles Images

Ce script permet de classifier de nouvelles images avec le modèle entraîné.

Utilisation :
    python predict_script.py chemin/vers/image.jpg
    python predict_script.py chemin/vers/dossier/

Compétences : Automatisation, Pipeline complet, Prédiction
"""

import numpy as np
from PIL import Image
import pickle
import sys
import os

MODEL_FILE = 'models/best_model.pkl'
IMAGE_SIZE = (32, 32)


def charger_modele():
    """
    Charge le modèle entraîné et ses composants.
    
    Returns:
        model: modèle entraîné
        scaler: StandardScaler
        classes_noms: noms des classes
    """
    if not os.path.exists(MODEL_FILE):
        print(f"ERREUR : Le fichier modele {MODEL_FILE} n'existe pas.")
        print("Veuillez d'abord entrainer un modele avec train_model_script.py")
        return None, None, None
    
    with open(MODEL_FILE, 'rb') as f:
        donnees = pickle.load(f)
    
    return donnees['model'], donnees['scaler'], donnees['classes_noms']


def charger_et_pretraiter_image(chemin_image):
    """
    Charge une image et la prépare pour la prédiction.
    
    Args:
        chemin_image: chemin vers l'image
    
    Returns:
        image_array: image prétraitée (normalisée)
    """
    try:
        # Ouvre l'image
        img = Image.open(chemin_image)
        
        # Convertit en RGB
        img = img.convert('RGB')
        
        # Redimensionne
        img = img.resize(IMAGE_SIZE)
        
        # Convertit en array et normalise
        img_array = np.array(img).astype('float32') / 255.0
        
        return img_array
    
    except Exception as e:
        print(f"Erreur lors du chargement de {chemin_image}: {e}")
        return None


def extraire_features_image(image):
    """
    Extrait les features d'une image (même méthode que dans feature_extraction_script.py).
    
    Args:
        image: array NumPy (hauteur, largeur, 3)
    
    Returns:
        features: vecteur 1D de features
    """
    features = []
    
    # 1. Histogramme de couleurs
    nb_bins = 16
    for canal in range(3):
        hist, _ = np.histogram(image[:, :, canal], bins=nb_bins, range=(0, 1))
        hist = hist / hist.sum()
        features.extend(hist)
    
    # 2. Statistiques par région
    grille = (4, 4)
    hauteur, largeur, canaux = image.shape
    nb_lignes, nb_colonnes = grille
    
    taille_h = hauteur // nb_lignes
    taille_w = largeur // nb_colonnes
    
    for i in range(nb_lignes):
        for j in range(nb_colonnes):
            region = image[i*taille_h:(i+1)*taille_h, 
                          j*taille_w:(j+1)*taille_w, :]
            
            for canal in range(canaux):
                region_canal = region[:, :, canal]
                moyenne = np.mean(region_canal)
                ecart_type = np.std(region_canal)
                features.extend([moyenne, ecart_type])
    
    return np.array(features)


def predire_image(chemin_image, model, scaler, classes_noms, afficher_proba=True):
    """
    Prédit la classe d'une image.
    
    Args:
        chemin_image: chemin vers l'image
        model: modèle entraîné
        scaler: StandardScaler
        classes_noms: noms des classes
        afficher_proba: si True, affiche les probabilités
    
    Returns:
        classe_predite: nom de la classe prédite
    """
    print(f"\nAnalyse de : {chemin_image}")
    
    # Charge et prétraite l'image
    image = charger_et_pretraiter_image(chemin_image)
    
    if image is None:
        return None
    
    # Extrait les features
    features = extraire_features_image(image)
    
    # Reshape pour le modèle (1 image = 1 ligne)
    features = features.reshape(1, -1)
    
    # Standardise
    features_scaled = scaler.transform(features)
    
    # Prédiction
    prediction = model.predict(features_scaled)[0]
    classe_predite = classes_noms[prediction]
    
    print(f"Classe predite : {classe_predite}")
    
    # Affiche les probabilités si disponible
    if afficher_proba and hasattr(model, 'predict_proba'):
        probas = model.predict_proba(features_scaled)[0]
        
        print("\nProbabilites par classe :")
        for i, classe in enumerate(classes_noms):
            print(f"  - {classe} : {probas[i]*100:.2f}%")
    
    return classe_predite


def predire_dossier(chemin_dossier, model, scaler, classes_noms):
    """
    Prédit les classes de toutes les images dans un dossier.
    
    Args:
        chemin_dossier: chemin vers le dossier
        model: modèle entraîné
        scaler: StandardScaler
        classes_noms: noms des classes
    """
    print(f"\nAnalyse du dossier : {chemin_dossier}")
    print("=" * 70)
    
    # Liste tous les fichiers images
    fichiers = [f for f in os.listdir(chemin_dossier) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if len(fichiers) == 0:
        print("Aucune image trouvee dans ce dossier.")
        return
    
    print(f"Nombre d'images trouvees : {len(fichiers)}\n")
    
    # Prédit chaque image
    resultats = []
    for fichier in fichiers:
        chemin_complet = os.path.join(chemin_dossier, fichier)
        classe = predire_image(chemin_complet, model, scaler, classes_noms, 
                              afficher_proba=False)
        
        if classe is not None:
            resultats.append((fichier, classe))
    
    # Résumé
    print("\n" + "=" * 70)
    print("RESUME DES PREDICTIONS")
    print("=" * 70)
    
    for fichier, classe in resultats:
        print(f"{fichier:<40} -> {classe}")


def mode_interactif(model, scaler, classes_noms):
    """
    Mode interactif pour prédire plusieurs images.
    """
    print("\n" + "=" * 70)
    print("MODE INTERACTIF")
    print("=" * 70)
    print("Entrez le chemin d'une image ou d'un dossier (ou 'q' pour quitter)")
    
    while True:
        chemin = input("\nChemin : ").strip()
        
        if chemin.lower() == 'q':
            print("Au revoir!")
            break
        
        if not os.path.exists(chemin):
            print(f"ERREUR : Le chemin {chemin} n'existe pas.")
            continue
        
        if os.path.isfile(chemin):
            predire_image(chemin, model, scaler, classes_noms)
        elif os.path.isdir(chemin):
            predire_dossier(chemin, model, scaler, classes_noms)
        else:
            print("Type de chemin non reconnu.")


def main():
    """
    Fonction principale pour la prédiction.
    """
    print("=" * 70)
    print("ÉTAPE 5 : PRÉDICTION SUR NOUVELLES IMAGES")
    print("=" * 70)
    
    # Charge le modèle
    print("\nChargement du modele...")
    model, scaler, classes_noms = charger_modele()
    
    if model is None:
        return
    
    print(f"Modele charge : {type(model).__name__}")
    print(f"Classes disponibles : {classes_noms}")
    
    # Vérifie les arguments de ligne de commande
    if len(sys.argv) > 1:
        chemin = sys.argv[1]
        
        if not os.path.exists(chemin):
            print(f"\nERREUR : Le chemin {chemin} n'existe pas.")
            return
        
        # Prédit une image ou un dossier
        if os.path.isfile(chemin):
            predire_image(chemin, model, scaler, classes_noms)
        elif os.path.isdir(chemin):
            predire_dossier(chemin, model, scaler, classes_noms)
    
    else:
        # Mode interactif si pas d'argument
        mode_interactif(model, scaler, classes_noms)


if __name__ == "__main__":
    main()
