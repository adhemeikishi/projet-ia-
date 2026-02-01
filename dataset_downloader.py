"""
Script 0 : Téléchargement et Préparation du Dataset (OPTIONNEL)

Ce script télécharge automatiquement un sous-ensemble du dataset CIFAR-10
et l'organise dans la structure attendue par le projet.

CIFAR-10 contient 10 classes : avion, voiture, oiseau, chat, cerf, chien,
grenouille, cheval, bateau, camion.

Pour ce projet, nous utiliserons seulement 3-5 classes pour simplifier.

Compétences : Automatisation, Préparation de données
"""

import os
import numpy as np
from PIL import Image
import pickle
import urllib.request
import tarfile

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_DIR = "data/raw"
TEMP_DIR = "temp"

# Classes CIFAR-10
CLASSES_CIFAR10 = [
    'avion', 'voiture', 'oiseau', 'chat', 'cerf',
    'chien', 'grenouille', 'cheval', 'bateau', 'camion'
]

# Classes à utiliser pour le projet (modifiable)
CLASSES_SELECTIONNEES = ['chat', 'chien', 'voiture']
NB_IMAGES_PAR_CLASSE = 500  # Nombre d'images à extraire par classe


def telecharger_cifar10():
    """
    Télécharge le dataset CIFAR-10 (environ 170 MB).
    
    Returns:
        chemin vers le fichier téléchargé
    """
    print("Telechargement du dataset CIFAR-10...")
    print("(Cela peut prendre quelques minutes selon votre connexion)")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    fichier_destination = os.path.join(TEMP_DIR, "cifar-10-python.tar.gz")
    
    if os.path.exists(fichier_destination):
        print("Le fichier existe deja, telechargement ignore.")
        return fichier_destination
    
    try:
        urllib.request.urlretrieve(CIFAR10_URL, fichier_destination)
        print(f"Telechargement termine : {fichier_destination}")
        return fichier_destination
    
    except Exception as e:
        print(f"Erreur lors du telechargement : {e}")
        return None


def extraire_cifar10(fichier_tar):
    """
    Extrait le fichier tar.gz de CIFAR-10.
    
    Args:
        fichier_tar: chemin vers le fichier tar.gz
    """
    print("\nExtraction du dataset...")
    
    try:
        with tarfile.open(fichier_tar, 'r:gz') as tar:
            tar.extractall(path=TEMP_DIR)
        
        print("Extraction terminee.")
    
    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")


def charger_batch_cifar10(fichier):
    """
    Charge un batch du dataset CIFAR-10.
    
    Args:
        fichier: chemin vers le fichier batch
    
    Returns:
        data: images (format numpy)
        labels: labels des images
    """
    with open(fichier, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    data = batch[b'data']
    labels = batch[b'labels']
    
    # Reshape les données (CIFAR-10 utilise un format spécifique)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return data, labels


def organiser_images_par_classe(classes_selectionnees, nb_images_par_classe):
    """
    Organise les images CIFAR-10 dans la structure du projet.
    
    Args:
        classes_selectionnees: liste des classes à extraire
        nb_images_par_classe: nombre d'images à extraire par classe
    """
    print("\nOrganisation des images par classe...")
    
    # Crée les dossiers pour chaque classe
    for classe in classes_selectionnees:
        dossier_classe = os.path.join(DATA_DIR, classe)
        os.makedirs(dossier_classe, exist_ok=True)
    
    # Charge les batches de CIFAR-10
    cifar_dir = os.path.join(TEMP_DIR, "cifar-10-batches-py")
    
    if not os.path.exists(cifar_dir):
        print(f"ERREUR : Le dossier {cifar_dir} n'existe pas.")
        return
    
    # Compteur d'images par classe
    compteur = {classe: 0 for classe in classes_selectionnees}
    
    # Indices des classes sélectionnées dans CIFAR-10
    indices_classes = [CLASSES_CIFAR10.index(c) for c in classes_selectionnees]
    
    # Parcourt les batches d'entraînement
    for i in range(1, 6):
        fichier_batch = os.path.join(cifar_dir, f"data_batch_{i}")
        
        print(f"Traitement du batch {i}/5...")
        
        data, labels = charger_batch_cifar10(fichier_batch)
        
        # Pour chaque image du batch
        for idx, (image, label) in enumerate(zip(data, labels)):
            
            # Vérifie si cette image appartient à une classe sélectionnée
            if label in indices_classes:
                nom_classe = CLASSES_CIFAR10[label]
                
                # Vérifie si on a assez d'images pour cette classe
                if compteur[nom_classe] < nb_images_par_classe:
                    # Sauvegarde l'image
                    nom_fichier = f"{nom_classe}_{compteur[nom_classe]:04d}.png"
                    chemin_image = os.path.join(DATA_DIR, nom_classe, nom_fichier)
                    
                    # Convertit en image PIL et sauvegarde
                    img = Image.fromarray(image)
                    img.save(chemin_image)
                    
                    compteur[nom_classe] += 1
        
        # Arrête si on a assez d'images pour toutes les classes
        if all(c >= nb_images_par_classe for c in compteur.values()):
            break
    
    # Affiche le résumé
    print("\n" + "=" * 70)
    print("RESUME DE L'ORGANISATION")
    print("=" * 70)
    for classe in classes_selectionnees:
        print(f"{classe} : {compteur[classe]} images")


def nettoyer_fichiers_temporaires():
    """
    Supprime les fichiers temporaires téléchargés.
    """
    reponse = input("\nVoulez-vous supprimer les fichiers temporaires ? (o/n) : ")
    
    if reponse.lower() == 'o':
        import shutil
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            print(f"Dossier {TEMP_DIR} supprime.")


def creer_dataset_personnalise():
    """
    Alternative : guide l'utilisateur pour créer son propre dataset.
    """
    print("\n" + "=" * 70)
    print("CREATION D'UN DATASET PERSONNALISE")
    print("=" * 70)
    print("\nVous pouvez creer votre propre dataset en suivant ces etapes :")
    print("\n1. Creez des dossiers pour chaque classe dans data/raw/")
    print("   Exemple :")
    print("     data/raw/chat/")
    print("     data/raw/chien/")
    print("     data/raw/voiture/")
    
    print("\n2. Ajoutez des images dans chaque dossier")
    print("   - Format : JPG, PNG, BMP")
    print("   - Minimum recommande : 100 images par classe")
    print("   - Les images seront automatiquement redimensionnees")
    
    print("\n3. Lancez le script preprocessing_script.py pour commencer")


def main():
    """
    Fonction principale pour le téléchargement du dataset.
    """
    print("=" * 70)
    print("ETAPE 0 : PREPARATION DU DATASET")
    print("=" * 70)
    
    print("\nOptions disponibles :")
    print("1. Telecharger et utiliser CIFAR-10 (recommande)")
    print("2. Creer un dataset personnalise")
    print("3. Quitter")
    
    choix = input("\nVotre choix (1/2/3) : ").strip()
    
    if choix == '1':
        # Télécharge CIFAR-10
        fichier_tar = telecharger_cifar10()
        
        if fichier_tar is None:
            return
        
        # Extrait le dataset
        extraire_cifar10(fichier_tar)
        
        # Organise les images
        print("\n" + "=" * 70)
        print("CONFIGURATION DU DATASET")
        print("=" * 70)
        print(f"Classes selectionnees : {CLASSES_SELECTIONNEES}")
        print(f"Nombre d'images par classe : {NB_IMAGES_PAR_CLASSE}")
        
        confirmer = input("\nConfirmer cette configuration ? (o/n) : ")
        
        if confirmer.lower() == 'o':
            organiser_images_par_classe(CLASSES_SELECTIONNEES, NB_IMAGES_PAR_CLASSE)
            
            print("\n" + "=" * 70)
            print("DATASET PRET")
            print("=" * 70)
            print(f"\nLes images ont ete organisees dans : {DATA_DIR}/")
            print("\nProchaine etape : executer preprocessing_script.py")
            
            # Nettoyage
            nettoyer_fichiers_temporaires()
    
    elif choix == '2':
        creer_dataset_personnalise()
    
    else:
        print("Au revoir!")


if __name__ == "__main__":
    main()
