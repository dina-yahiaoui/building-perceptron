"""
script.py
Implémentation d'un Perceptron (Rosenblatt, 1958) en programmation orientée objet.
Projet Building Perceptron - La Plateforme
"""

import numpy as np


class Perceptron:
    """Perceptron simple pour la classification binaire (classes 0 et 1)."""

    def __init__(self, learning_rate=0.01, n_epochs=100, random_state=42):
        # learning_rate (η) : taille des corrections appliquées aux poids
        # n_epochs : nombre de passages complets sur les données
        # random_state : graine aléatoire, pour avoir les mêmes résultats à chaque exécution
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights = None   # w : un poids par variable
        self.bias = None      # b : le biais
        self.errors_ = []     # nombre d'erreurs à chaque epoch (pour tracer la courbe d'apprentissage)

    def _activation(self, z):
        """Fonction de Heaviside : 1 si z >= 0, sinon 0."""
        return np.where(z >= 0, 1, 0)

    def net_input(self, X):
        """Somme pondérée : z = w · x + b"""
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        """Entraîne le perceptron sur X (n_exemples, n_variables) et y (0 ou 1)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        # 1. Initialisation : poids à 0
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.errors_ = []

        # 2. On répète sur plusieurs epochs
        for _ in range(self.n_epochs):
            errors = 0
            # On mélange l'ordre des exemples à chaque epoch
            for i in rng.permutation(len(X)):
                xi, target = X[i], y[i]
                prediction = self._activation(self.net_input(xi))

                # 3. Règle du perceptron : w <- w + η (y - ŷ) x
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
                self.bias += update

                errors += int(update != 0)

            self.errors_.append(errors)

            # 4. Arrêt anticipé : plus aucune erreur, le perceptron a convergé
            if errors == 0:
                break

        return self

    def predict(self, X):
        """Prédit la classe (0 ou 1) de chaque exemple."""
        X = np.asarray(X, dtype=float)
        return self._activation(self.net_input(X))

    def score(self, X, y):
        """Accuracy : proportion de bonnes prédictions."""
        return np.mean(self.predict(X) == np.asarray(y))