"""
Script 1 : Chargement et Prétraitement des images

Ce script va :
1. Charger les images depuis le dossier data/raw/
2. Les redimensionner à une taille fixe
3. Les normaliser (valeurs entre 0 et 1)
4. Sauvegarder les données prétraitées

Compétences : Python, NumPy, Prétraitement de données
"""

import os
import numpy as np
from PIL import Image
import pickle

# Configuration
IMAGE_SIZE = (32, 32)  # Taille fixe pour toutes les images
DATA_DIR = 'data/raw'  # Dossier contenant les sous-dossiers de classes
OUTPUT_DIR = 'data/processed'
OUTPUT_FILE = 'preprocessed_data.pkl'

def charger_images_depuis_dossier(chemin_dossier, label):
    """
    Charge toutes les images d'un dossier et leur attribue un label.
    
    Args:
        chemin_dossier: chemin vers le dossier contenant les images
        label: étiquette de la classe (ex: 'chat', 'chien')
    
    Returns:
        images: liste des images chargées
        labels: liste des labels correspondants
    """
    images = []
    labels = []
    
    # Liste tous les fichiers dans le dossier
    fichiers = os.listdir(chemin_dossier)
    
    for nom_fichier in fichiers:
        # Vérifie que c'est bien une image
        if nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            chemin_complet = os.path.join(chemin_dossier, nom_fichier)
            
            try:
                # Ouvre l'image avec PIL
                img = Image.open(chemin_complet)
                
                # Convertit en RGB (au cas où l'image est en niveaux de gris)
                img = img.convert('RGB')
                
                # Redimensionne l'image à la taille fixe
                img = img.resize(IMAGE_SIZE)
                
                # Convertit en array NumPy
                img_array = np.array(img)
                
                images.append(img_array)
                labels.append(label)
                
            except Exception as e:
                print(f"Erreur lors du chargement de {nom_fichier}: {e}")
    
    return images, labels


def normaliser_images(images):
    """
    Normalise les valeurs des pixels entre 0 et 1.
    
    Les pixels ont des valeurs entre 0 et 255.
    La normalisation facilite l'apprentissage des modèles.
    
    Args:
        images: liste d'arrays NumPy
    
    Returns:
        images_normalisees: array NumPy normalisé
    """
    # Convertit la liste en array NumPy
    images_array = np.array(images)
    
    # Divise par 255 pour obtenir des valeurs entre 0 et 1
    images_normalisees = images_array.astype('float32') / 255.0
    
    return images_normalisees


def charger_toutes_les_classes(data_dir):
    """
    Charge toutes les images de toutes les classes.
    
    Structure attendue :
    data/raw/
        classe_1/
            image1.jpg
            image2.jpg
        classe_2/
            image1.jpg
    
    Returns:
        X: array NumPy des images (n_images, hauteur, largeur, 3)
        y: array NumPy des labels
        classes_noms: liste des noms de classes
    """
    toutes_images = []
    tous_labels = []
    classes_noms = []
    
    # Liste tous les sous-dossiers (chaque sous-dossier = une classe)
    classes = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Classes detectees : {classes}")
    
    # Pour chaque classe
    for idx, nom_classe in enumerate(sorted(classes)):
        chemin_classe = os.path.join(data_dir, nom_classe)
        
        print(f"\nChargement de la classe '{nom_classe}'...")
        images, labels = charger_images_depuis_dossier(chemin_classe, idx)
        
        print(f"  -> {len(images)} images chargees")
        
        toutes_images.extend(images)
        tous_labels.extend(labels)
        classes_noms.append(nom_classe)
    
    # Normalise les images
    print("\nNormalisation des images...")
    X = normaliser_images(toutes_images)
    y = np.array(tous_labels)
    
    return X, y, classes_noms


def sauvegarder_donnees(X, y, classes_noms, output_dir, output_file):
    """
    Sauvegarde les données prétraitées dans un fichier pickle.
    
    Args:
        X: images prétraitées
        y: labels
        classes_noms: noms des classes
        output_dir: dossier de sortie
        output_file: nom du fichier
    """
    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    chemin_complet = os.path.join(output_dir, output_file)
    
    # Sauvegarde dans un dictionnaire
    donnees = {
        'X': X,
        'y': y,
        'classes_noms': classes_noms,
        'image_size': IMAGE_SIZE
    }
    
    with open(chemin_complet, 'wb') as f:
        pickle.dump(donnees, f)
    
    print(f"\nDonnees sauvegardees dans : {chemin_complet}")


def main():
    """
    Fonction principale qui orchestre le prétraitement.
    """
    print("=" * 50)
    print("ÉTAPE 1 : PRÉTRAITEMENT DES IMAGES")
    print("=" * 50)
    
    # Vérifie que le dossier existe
    if not os.path.exists(DATA_DIR):
        print(f"\nERREUR : Le dossier {DATA_DIR} n'existe pas.")
        print("Veuillez creer la structure suivante :")
        print("data/raw/")
        print("    classe_1/")
        print("    classe_2/")
        print("    classe_3/")
        return
    
    # Charge toutes les images
    X, y, classes_noms = charger_toutes_les_classes(DATA_DIR)
    
    # Affiche les statistiques
    print("\n" + "=" * 50)
    print("STATISTIQUES DES DONNEES")
    print("=" * 50)
    print(f"Nombre total d'images : {len(X)}")
    print(f"Forme des images : {X.shape}")
    print(f"Nombre de classes : {len(classes_noms)}")
    print(f"Classes : {classes_noms}")
    print(f"\nRepartition par classe :")
    for idx, nom_classe in enumerate(classes_noms):
        nb_images = np.sum(y == idx)
        print(f"  - {nom_classe} : {nb_images} images")
    
    # Sauvegarde les données
    sauvegarder_donnees(X, y, classes_noms, OUTPUT_DIR, OUTPUT_FILE)
    
    print("\n" + "=" * 50)
    print("PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 50)


if __name__ == "__main__":
    main()
