"""
Script 3 : Entraînement du Modèle de Classification

Ce script :
1. Sépare les données en ensembles d'entraînement et de test
2. Entraîne plusieurs modèles classiques (KNN, SVM, Régression Logistique)
3. Compare leurs performances
4. Sauvegarde le meilleur modèle

Compétences : Scikit-Learn, Machine Learning, Classification
"""

import numpy as np
import pickle
import os
import time

# Essayer d'importer scikit-learn. Si absent (ex: environnement Python non compatible),
# on fournit un fallback minimal (SimpleKNN) pour permettre l'exécution du pipeline.
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except Exception:
    print("Avertissement : scikit-learn n'est pas disponible dans cet environnement.")
    print("Les modèles SVM et Logistic Regression seront ignorés. KNN fallback sera utilisé.")
    SKLEARN_AVAILABLE = False

    # Fallback minimal pour certains utilitaires
    def accuracy_score(y_true, y_pred):
        return float((y_true == y_pred).mean())

    def classification_report(y_true, y_pred, target_names=None, digits=4):
        return "Classification report non disponible sans scikit-learn."

    class KNeighborsClassifier:
        def __init__(self, n_neighbors=5):
            self.n_neighbors = n_neighbors

        def fit(self, X, y):
            self.X = X
            self.y = y

        def predict(self, X_test):
            preds = []
            for x in X_test:
                dists = np.linalg.norm(self.X - x, axis=1)
                idx = np.argsort(dists)[:self.n_neighbors]
                vals = self.y[idx]
                values, counts = np.unique(vals, return_counts=True)
                preds.append(values[np.argmax(counts)])
            return np.array(preds)

    # SVM and LogisticRegression will be unavailable in fallback (skipped)

    # Fallback pour train_test_split (supporte stratify)
    def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
        rng = np.random.RandomState(random_state)
        X = np.array(X)
        y = np.array(y)
        n = len(X)
        if stratify is not None:
            # Stratified split: on split by class to keep proportions
            train_idx = []
            test_idx = []
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                rng.shuffle(cls_idx)
                n_test = max(1, int(len(cls_idx) * test_size))
                test_idx.extend(cls_idx[:n_test].tolist())
                train_idx.extend(cls_idx[n_test:].tolist())
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        else:
            indices = np.arange(n)
            rng.shuffle(indices)
            n_test = int(n * test_size)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    # Fallback minimal pour StandardScaler
    class StandardScaler:
        def fit(self, X):
            X = np.array(X)
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            return self
        def fit_transform(self, X):
            self.fit(X)
            return (X - self.mean_) / self.scale_
        def transform(self, X):
            return (X - self.mean_) / self.scale_


INPUT_FILE = 'data/processed/features.pkl'
MODELS_DIR = 'models'
TEST_SIZE = 0.2  # 20% des données pour le test
RANDOM_STATE = 42  # Pour la reproductibilité


def charger_features():
    """
    Charge les features extraites précédemment.
    
    Returns:
        X_features: features des images
        y: labels
        classes_noms: noms des classes
    """
    print(f"Chargement des features depuis {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERREUR : Le fichier {INPUT_FILE} n'existe pas.")
        print("Veuillez d'abord executer feature_extraction_script.py")
        return None, None, None
    
    with open(INPUT_FILE, 'rb') as f:
        donnees = pickle.load(f)
    
    return donnees['X_features'], donnees['y'], donnees['classes_noms']


def separer_train_test(X, y, test_size=0.2, random_state=42):
    """
    Sépare les données en ensembles d'entraînement et de test.
    
    Le test set permet d'évaluer le modèle sur des données qu'il n'a jamais vues.
    C'est crucial pour mesurer sa capacité de généralisation.
    
    Args:
        X: features
        y: labels
        test_size: proportion de données pour le test (0.2 = 20%)
        random_state: graine aléatoire pour reproductibilité
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\nSéparation des données : {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Garde les mêmes proportions de classes
    )
    
    print(f"  -> Ensemble d'entrainement : {len(X_train)} images")
    print(f"  -> Ensemble de test : {len(X_test)} images")
    
    return X_train, X_test, y_train, y_test


def standardiser_features(X_train, X_test):
    """
    Standardise les features (moyenne = 0, écart-type = 1).
    
    La standardisation est importante pour :
    - KNN : les distances entre points doivent être comparables
    - SVM : améliore la convergence et les performances
    - Régression Logistique : améliore la convergence
    
    IMPORTANT : On fit le scaler sur le train set uniquement,
    puis on l'applique au test set pour éviter la fuite de données.
    
    Args:
        X_train: features d'entraînement
        X_test: features de test
    
    Returns:
        X_train_scaled: features d'entraînement standardisées
        X_test_scaled: features de test standardisées
        scaler: objet StandardScaler (à sauvegarder pour la prédiction)
    """
    print("\nStandardisation des features...")
    
    # Crée le scaler et le fit sur le train set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Applique le même scaler au test set
    X_test_scaled = scaler.transform(X_test)
    
    print("  -> Standardisation terminee")
    
    return X_train_scaled, X_test_scaled, scaler


def entrainer_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    """
    Entraîne un modèle K-Nearest Neighbors.
    
    Principe : Pour classifier une nouvelle image, on regarde ses k plus proches
    voisins dans l'espace des features, et on prend la classe majoritaire.
    
    Args:
        X_train, y_train: données d'entraînement
        X_test, y_test: données de test
        n_neighbors: nombre de voisins à considérer
    
    Returns:
        model: modèle entraîné
        accuracy: précision sur le test set
        temps: temps d'entraînement
    """
    print(f"\nEntraînement du modèle KNN (k={n_neighbors})...")
    
    debut = time.time()
    
    # Crée et entraîne le modèle
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    temps = time.time() - debut
    
    # Évalue sur le test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  -> Entrainement termine en {temps:.2f} secondes")
    print(f"  -> Accuracy sur le test set : {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return model, accuracy, temps


def entrainer_svm(X_train, y_train, X_test, y_test, kernel='rbf', C=1.0):
    """
    Entraîne un modèle Support Vector Machine.
    
    Principe : Trouve l'hyperplan qui sépare au mieux les classes
    en maximisant la marge entre les classes.
    
    Args:
        X_train, y_train: données d'entraînement
        X_test, y_test: données de test
        kernel: type de kernel ('rbf', 'linear', 'poly')
        C: paramètre de régularisation
    
    Returns:
        model: modèle entraîné
        accuracy: précision sur le test set
        temps: temps d'entraînement
    """
    print(f"\nEntrainement du modele SVM (kernel={kernel}, C={C})...")
    
    debut = time.time()
    
    # Crée et entraîne le modèle
    model = SVC(kernel=kernel, C=C, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    temps = time.time() - debut
    
    # Évalue sur le test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  -> Entrainement termine en {temps:.2f} secondes")
    print(f"  -> Accuracy sur le test set : {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return model, accuracy, temps


def entrainer_regression_logistique(X_train, y_train, X_test, y_test, max_iter=1000):
    """
    Entraîne un modèle de Régression Logistique.
    
    Principe : Modèle linéaire qui calcule des probabilités d'appartenance
    à chaque classe. Simple et rapide, mais moins performant sur images complexes.
    
    Args:
        X_train, y_train: données d'entraînement
        X_test, y_test: données de test
        max_iter: nombre maximum d'itérations
    
    Returns:
        model: modèle entraîné
        accuracy: précision sur le test set
        temps: temps d'entraînement
    """
    print(f"\nEntrainement du modele Regression Logistique...")
    
    debut = time.time()
    
    # Crée et entraîne le modèle
    model = LogisticRegression(max_iter=max_iter, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    temps = time.time() - debut
    
    # Évalue sur le test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  -> Entrainement termine en {temps:.2f} secondes")
    print(f"  -> Accuracy sur le test set : {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return model, accuracy, temps


def comparer_modeles(resultats):
    """
    Affiche un tableau comparatif des modèles.
    
    Args:
        resultats: liste de tuples (nom, model, accuracy, temps)
    
    Returns:
        meilleur_nom, meilleur_model: le modèle avec la meilleure accuracy
    """
    print("\n" + "=" * 70)
    print("COMPARAISON DES MODELES")
    print("=" * 70)
    print(f"{'Modele':<25} {'Accuracy':<15} {'Temps (s)':<15}")
    print("-" * 70)
    
    meilleur_accuracy = 0
    meilleur_nom = None
    meilleur_model = None
    
    for nom, model, accuracy, temps in resultats:
        print(f"{nom:<25} {accuracy*100:>6.2f}%         {temps:>8.2f}")
        
        if accuracy > meilleur_accuracy:
            meilleur_accuracy = accuracy
            meilleur_nom = nom
            meilleur_model = model
    
    print("-" * 70)
    print(f"\nMeilleur modele : {meilleur_nom} ({meilleur_accuracy*100:.2f}%)")
    
    return meilleur_nom, meilleur_model


def sauvegarder_modele(model, scaler, classes_noms, nom_modele):
    """
    Sauvegarde le modèle entraîné et le scaler.
    
    Args:
        model: modèle entraîné
        scaler: StandardScaler utilisé
        classes_noms: noms des classes
        nom_modele: nom du fichier
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Sauvegarde tout dans un dictionnaire
    donnees_modele = {
        'model': model,
        'scaler': scaler,
        'classes_noms': classes_noms
    }
    
    chemin = os.path.join(MODELS_DIR, f'{nom_modele}.pkl')
    
    with open(chemin, 'wb') as f:
        pickle.dump(donnees_modele, f)
    
    print(f"\nModele sauvegarde dans : {chemin}")


def main():
    """
    Fonction principale pour l'entraînement des modèles.
    """
    print("=" * 70)
    print("ÉTAPE 3 : ENTRAÎNEMENT DES MODÈLES")
    print("=" * 70)
    
    # Charge les features
    X_features, y, classes_noms = charger_features()
    
    if X_features is None:
        return
    
    print(f"\nDonnees chargees : {X_features.shape[0]} images, {X_features.shape[1]} features")
    print(f"Classes : {classes_noms}")
    
    # Sépare train/test
    print("\n" + "=" * 70)
    print("PREPARATION DES DONNEES")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = separer_train_test(
        X_features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Standardise les features
    X_train_scaled, X_test_scaled, scaler = standardiser_features(X_train, X_test)
    
    # Entraîne plusieurs modèles
    print("\n" + "=" * 70)
    print("ENTRAINEMENT DES MODELES")
    print("=" * 70)
    
    resultats = []
    
    # 1. KNN avec k=5 (toujours disponible, fallback si nécessaire)
    knn_model, knn_acc, knn_temps = entrainer_knn(
        X_train_scaled, y_train, X_test_scaled, y_test, n_neighbors=5
    )
    resultats.append(("KNN (k=5)", knn_model, knn_acc, knn_temps))
    
    if SKLEARN_AVAILABLE:
        # 2. SVM avec kernel RBF
        svm_model, svm_acc, svm_temps = entrainer_svm(
            X_train_scaled, y_train, X_test_scaled, y_test, kernel='rbf', C=1.0
        )
        resultats.append(("SVM (RBF)", svm_model, svm_acc, svm_temps))
        
        # 3. Régression Logistique
        lr_model, lr_acc, lr_temps = entrainer_regression_logistique(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        resultats.append(("Regression Logistique", lr_model, lr_acc, lr_temps))
    else:
        print("\nscikit-learn n'est pas disponible: SVM et Regression Logistique ignores, seul KNN a ete entraine.")
    
    # Compare les modèles
    meilleur_nom, meilleur_model = comparer_modeles(resultats)
    
    # Sauvegarde le meilleur modèle
    print("\n" + "=" * 70)
    print("SAUVEGARDE DU MEILLEUR MODELE")
    print("=" * 70)
    
    sauvegarder_modele(meilleur_model, scaler, classes_noms, 'best_model')
    
    # Sauvegarde aussi tous les modèles pour comparaison
    for nom, model, _, _ in resultats:
        nom_fichier = nom.lower().replace(' ', '_').replace('(', '').replace(')', '')
        sauvegarder_modele(model, scaler, classes_noms, nom_fichier)
    
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 70)
    print(f"\nProchaine étape : exécuter evaluate_script.py pour analyse détaillée")


if __name__ == "__main__":
    main()
