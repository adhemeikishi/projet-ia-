"""
Fonctions utilitaires réutilisables pour le projet : I/O, helpers, checks.
"""
import os
import pickle


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_pickle(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def file_exists(path):
    return os.path.exists(path)
