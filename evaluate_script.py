"""
Script 4 : Évaluation Détaillée du Modèle

Ce script analyse en profondeur les performances du modèle :
1. Matrice de confusion
2. Rapport de classification (précision, rappel, F1-score)
3. Visualisations des performances
4. Analyse des erreurs

Compétences : Scikit-Learn, Pandas, Matplotlib, Métriques d'évaluation
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Import des métriques : si scikit-learn est absent, on fournit des implémentations de secours
try:
    from sklearn.metrics import (
        confusion_matrix, 
        classification_report, 
        accuracy_score,
        precision_recall_fscore_support
    )
except Exception:
    print("Avertissement : scikit-learn non disponible. Utilisation de métriques de secours.")
    def confusion_matrix(y_true, y_pred):
        import numpy as _np
        labels = _np.unique(_np.concatenate([y_true, y_pred]))
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        cm = _np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[label_to_idx[t], label_to_idx[p]] += 1
        return cm

    def classification_report(y_true, y_pred, target_names=None, digits=4):
        return "Classification report non disponible sans scikit-learn."

    def accuracy_score(y_true, y_pred):
        import numpy as _np
        return float((_np.array(y_true) == _np.array(y_pred)).mean())

    def precision_recall_fscore_support(y_true, y_pred, average=None):
        import numpy as _np
        labels = _np.unique(_np.concatenate([y_true, y_pred]))
        precision = []
        recall = []
        f1 = []
        support = []
        for lbl in labels:
            tp = int(_np.sum((_np.array(y_true) == lbl) & (_np.array(y_pred) == lbl)))
            fp = int(_np.sum((_np.array(y_true) != lbl) & (_np.array(y_pred) == lbl)))
            fn = int(_np.sum((_np.array(y_true) == lbl) & (_np.array(y_pred) != lbl)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            sup = int(_np.sum(_np.array(y_true) == lbl))
            precision.append(prec)
            recall.append(rec)
            f1.append(f)
            support.append(sup)
        return _np.array(precision), _np.array(recall), _np.array(f1), _np.array(support)

INPUT_FEATURES = 'data/processed/features.pkl'
INPUT_MODEL = 'models/best_model.pkl'
OUTPUT_DIR = 'results'


def charger_donnees():
    """
    Charge les features et le modèle entraîné.
    
    Returns:
        X, y: données complètes
        model: modèle entraîné
        scaler: StandardScaler
        classes_noms: noms des classes
    """
    print("Chargement des donnees et du modele...")
    
    # Charge les features
    with open(INPUT_FEATURES, 'rb') as f:
        donnees_features = pickle.load(f)
    
    X = donnees_features['X_features']
    y = donnees_features['y']
    
    # Charge le modèle
    with open(INPUT_MODEL, 'rb') as f:
        donnees_modele = pickle.load(f)
    
    model = donnees_modele['model']
    scaler = donnees_modele['scaler']
    classes_noms = donnees_modele['classes_noms']
    
    print(f"  -> {len(X)} images chargees")
    print(f"  -> Modele : {type(model).__name__}")
    
    return X, y, model, scaler, classes_noms


def calculer_predictions(model, scaler, X, y):
    """
    Calcule les prédictions du modèle sur toutes les données.
    
    Args:
        model: modèle entraîné
        scaler: StandardScaler
        X: features
        y: vrais labels
    
    Returns:
        y_pred: prédictions du modèle
    """
    print("\nCalcul des predictions...")
    
    # Standardise les features
    X_scaled = scaler.transform(X)
    
    # Prédictions
    y_pred = model.predict(X_scaled)
    
    # Calcule l'accuracy globale
    accuracy = accuracy_score(y, y_pred)
    print(f"  -> Accuracy globale : {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return y_pred


def afficher_matrice_confusion(y_true, y_pred, classes_noms):
    """
    Calcule et affiche la matrice de confusion.
    
    La matrice de confusion montre :
    - En diagonale : prédictions correctes
    - Hors diagonale : confusions entre classes
    
    Args:
        y_true: vrais labels
        y_pred: prédictions
        classes_noms: noms des classes
    """
    print("\n" + "=" * 70)
    print("MATRICE DE CONFUSION")
    print("=" * 70)
    
    # Calcule la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Crée un DataFrame pour un affichage plus lisible
    df_cm = pd.DataFrame(
        cm, 
        index=classes_noms, 
        columns=classes_noms
    )
    
    print("\nMatrice de confusion (lignes = verite, colonnes = prediction) :")
    print(df_cm)
    
    # Visualisation
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Predite')
    plt.tight_layout()
    
    # Sauvegarde
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'matrice_confusion.png'))
    print(f"\nGraphique sauvegarde : {OUTPUT_DIR}/matrice_confusion.png")
    
    return cm


def afficher_rapport_classification(y_true, y_pred, classes_noms):
    """
    Affiche le rapport de classification détaillé.
    
    Métriques par classe :
    - Precision : parmi les prédictions positives, combien sont correctes
    - Recall : parmi les vrais positifs, combien ont été détectés
    - F1-Score : moyenne harmonique de précision et rappel
    - Support : nombre d'exemples de cette classe
    
    Args:
        y_true: vrais labels
        y_pred: prédictions
        classes_noms: noms des classes
    """
    print("\n" + "=" * 70)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 70)
    
    # Génère le rapport
    rapport = classification_report(
        y_true, 
        y_pred, 
        target_names=classes_noms,
        digits=4
    )
    
    print("\n" + rapport)
    
    # Calcule les métriques manuellement pour un affichage personnalisé
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Crée un DataFrame
    df_metriques = pd.DataFrame({
        'Classe': classes_noms,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Visualisation des métriques par classe
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Précision
    axes[0].bar(classes_noms, precision, color='skyblue')
    axes[0].set_title('Precision par Classe')
    axes[0].set_ylabel('Precision')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[1].bar(classes_noms, recall, color='lightgreen')
    axes[1].set_title('Recall par Classe')
    axes[1].set_ylabel('Recall')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    
    # F1-Score
    axes[2].bar(classes_noms, f1, color='lightcoral')
    axes[2].set_title('F1-Score par Classe')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_ylim([0, 1])
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metriques_par_classe.png'))
    print(f"Graphique sauvegarde : {OUTPUT_DIR}/metriques_par_classe.png")
    
    return df_metriques


def analyser_erreurs(y_true, y_pred, classes_noms):
    """
    Analyse détaillée des erreurs de classification.
    
    Args:
        y_true: vrais labels
        y_pred: prédictions
        classes_noms: noms des classes
    """
    print("\n" + "=" * 70)
    print("ANALYSE DES ERREURS")
    print("=" * 70)
    
    # Identifie les erreurs
    erreurs = y_true != y_pred
    nb_erreurs = np.sum(erreurs)
    nb_total = len(y_true)
    
    print(f"\nNombre total d'erreurs : {nb_erreurs} / {nb_total}")
    print(f"Taux d'erreur : {(nb_erreurs/nb_total)*100:.2f}%")
    
    # Analyse des confusions les plus fréquentes
    print("\nConfusions les plus frequentes :")
    
    confusions = []
    for i in range(len(classes_noms)):
        for j in range(len(classes_noms)):
            if i != j:
                nb_confusions = np.sum((y_true == i) & (y_pred == j))
                if nb_confusions > 0:
                    confusions.append((
                        classes_noms[i], 
                        classes_noms[j], 
                        nb_confusions
                    ))
    
    # Trie par nombre de confusions
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    # Affiche les 5 confusions les plus fréquentes
    for vraie_classe, classe_predite, nb in confusions[:5]:
        print(f"  - {vraie_classe} confondu avec {classe_predite} : {nb} fois")


def generer_rapport_complet(y_true, y_pred, classes_noms, df_metriques):
    """
    Génère un rapport complet en fichier texte.
    
    Args:
        y_true: vrais labels
        y_pred: prédictions
        classes_noms: noms des classes
        df_metriques: DataFrame des métriques
    """
    rapport_path = os.path.join(OUTPUT_DIR, 'rapport_evaluation.txt')
    
    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPPORT D'EVALUATION DU MODELE\n")
        f.write("=" * 70 + "\n\n")
        
        # Accuracy globale
        accuracy = accuracy_score(y_true, y_pred)
        f.write(f"Accuracy globale : {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        # Métriques par classe
        f.write("Metriques par classe :\n")
        f.write("-" * 70 + "\n")
        f.write(df_metriques.to_string(index=False))
        f.write("\n\n")
        
        # Rapport de classification complet
        f.write("Rapport de classification detaille :\n")
        f.write("-" * 70 + "\n")
        rapport = classification_report(y_true, y_pred, target_names=classes_noms)
        f.write(rapport)
    
    print(f"\nRapport complet sauvegarde : {rapport_path}")


def main():
    """
    Fonction principale pour l'évaluation.
    """
    print("=" * 70)
    print("ÉTAPE 4 : ÉVALUATION DÉTAILLÉE DU MODÈLE")
    print("=" * 70)
    
    # Charge les données et le modèle
    X, y, model, scaler, classes_noms = charger_donnees()
    
    # Calcule les prédictions
    y_pred = calculer_predictions(model, scaler, X, y)
    
    # Matrice de confusion
    cm = afficher_matrice_confusion(y, y_pred, classes_noms)
    
    # Rapport de classification
    df_metriques = afficher_rapport_classification(y, y_pred, classes_noms)
    
    # Analyse des erreurs
    analyser_erreurs(y, y_pred, classes_noms)
    
    # Génère un rapport complet
    print("\n" + "=" * 70)
    print("GENERATION DU RAPPORT")
    print("=" * 70)
    generer_rapport_complet(y, y_pred, classes_noms, df_metriques)
    
    print("\n" + "=" * 70)
    print("ÉVALUATION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)
    print(f"\nTous les resultats sont disponibles dans le dossier : {OUTPUT_DIR}/")
    print(f"\nProchaine etape : executer predict_script.py pour classifier de nouvelles images")


if __name__ == "__main__":
    main()
