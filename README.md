# Building Perceptron

Implémentation d'un Perceptron en Python (programmation orientée objet), appliqué à la prédiction du caractère bénin ou malin de tumeurs du sein à partir du jeu de données *Breast Cancer Wisconsin (Diagnostic)*.

Projet réalisé dans le cadre du Bachelor IA / Data de La Plateforme (Marseille).

---

## Sommaire

1. [Contexte du projet](#1-contexte-du-projet)
2. [Les données et leur analyse](#2-les-données-et-leur-analyse)
3. [Méthodologie](#3-méthodologie)
4. [Résultats](#4-résultats)
5. [Outils utilisés](#5-outils-utilisés)
6. [Structure du dépôt et installation](#6-structure-du-dépôt-et-installation)
7. [Conclusion](#7-conclusion)
8. [Bibliographie](#8-bibliographie)

---

## 1. Contexte du projet

### Machine Learning et Deep Learning

Le **Machine Learning** regroupe les méthodes qui permettent à un programme d'apprendre une règle à partir d'exemples, sans qu'elle soit codée à la main. Le **Deep Learning** en est une sous-famille, fondée sur des réseaux de neurones à plusieurs couches capables d'extraire eux-mêmes les caractéristiques utiles de données brutes (images, son, texte).

### Le Perceptron

Le Perceptron, proposé par **Frank Rosenblatt en 1958**, est le premier neurone artificiel capable d'apprendre. Il est l'ancêtre des réseaux de neurones actuels.

Inspiré du neurone biologique, il reçoit des entrées, les pondère, les additionne et décide de « s'activer » ou non :

```
ŷ = 1  si  w · x + b ≥ 0
ŷ = 0  sinon
```

| Terme | Signification |
|---|---|
| `x` | Entrées (les variables d'un exemple) |
| `w` | Poids, appris pendant l'entraînement |
| `b` | Biais, qui décale le seuil de décision |
| `ŷ` | Classe prédite (0 ou 1) |

À chaque erreur, les poids sont corrigés selon la **règle du Perceptron** :

```
w ← w + η (y − ŷ) x
b ← b + η (y − ŷ)
```

où `η` est le taux d'apprentissage. Si la prédiction est juste, `y − ŷ = 0` et rien ne change.

**Limite principale :** le Perceptron ne peut tracer qu'une frontière linéaire. Il ne converge que si les données sont linéairement séparables.

### Objectif

1. Développer une classe `Perceptron` en Python et la valider sur des données factices.
2. L'appliquer à un problème de santé réel : prédire si une tumeur du sein est **bénigne** ou **maligne**.
3. Évaluer son efficacité et proposer des alternatives.

---

## 2. Les données et leur analyse

### Source

Jeu de données **Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository, disponible sur [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

Chaque ligne décrit une tumeur à partir d'une image numérisée de cytoponction (ponction à l'aiguille fine). Dix caractéristiques des noyaux cellulaires sont mesurées (rayon, texture, périmètre, aire, lissage, compacité, concavité, points concaves, symétrie, dimension fractale), chacune résumée de trois façons : moyenne (`_mean`), erreur standard (`_se`) et pire valeur (`_worst`).

| Caractéristique | Valeur |
|---|---|
| Nombre de tumeurs | 569 |
| Variables explicatives | 30 (numériques) |
| Variable cible | `diagnosis` : M (malin) / B (bénin) |

### Nettoyage

| Problème détecté | Traitement |
|---|---|
| Colonne `Unnamed: 32` entièrement vide (artefact d'export) | Supprimée |
| Colonne `id` sans valeur prédictive | Supprimée |
| Cible au format texte | Encodée : M = 1, B = 0 |
| 13 tumeurs avec des valeurs nulles de concavité | Conservées : contour lisse, pas des données manquantes |
| 171 tumeurs avec au moins une valeur extrême (règle de l'écart interquartile) | Conservées : vraies mesures, souvent des tumeurs malignes |

Après nettoyage : **569 tumeurs × 30 variables**, aucune valeur manquante, aucun doublon.

### Principaux enseignements de l'analyse exploratoire

- **Classes déséquilibrées :** 357 bénignes (63 %) et 212 malignes (37 %). L'accuracy seule ne suffit donc pas pour évaluer le modèle.
- **Variables discriminantes :** la taille (`radius`, `perimeter`, `area`) et l'irrégularité du contour (`concavity`, `concave points`) séparent nettement les deux classes. Variable la plus corrélée au diagnostic : `concave points_worst` (r = 0,79).
- **Échelles très différentes :** de moins de 0,2 (`smoothness_mean`) à plus de 2 500 (`area_mean`), ce qui impose une standardisation.
- **Forte redondance :** 21 paires de variables corrélées à plus de 0,9 (par exemple `radius_mean` et `perimeter_mean`, r = 0,998), ce qui justifie une réduction de dimensionnalité.

---

## 3. Méthodologie

1. **Séparation entraînement / test** : 80 % / 20 % (455 / 114 tumeurs), stratifiée pour conserver 37 % de tumeurs malignes dans chaque jeu.
2. **Standardisation** (`StandardScaler`), ajustée sur le jeu d'entraînement uniquement pour éviter toute fuite de données.
3. **Réduction de dimensionnalité par ACP** : 10 composantes principales conservent **95,2 %** de la variance. On passe ainsi de 30 à 10 variables non corrélées.
   - PC1 (44,6 % de la variance) : concavité, compacité et taille ; elle traduit la « gravité morphologique ».
   - PC2 (18,5 %) : dimension fractale du contour.
4. **Modélisation** avec la classe `Perceptron` développée dans `script.py`.
5. **Évaluation** sur le jeu de test avec des métriques adaptées au contexte médical.

---

## 4. Résultats

### Validation sur données factices

Sur 200 points générés aléatoirement en deux groupes linéairement séparables, le Perceptron converge (0 erreur) et obtient **100 % d'accuracy** sur le jeu de test.

### Application au cancer du sein

| Métrique | Valeur | Signification |
|---|---|---|
| Accuracy | 97,4 % | Part de diagnostics corrects |
| Précision | 95,3 % | Parmi les tumeurs prédites malignes, part réellement malignes |
| **Rappel** | **97,6 %** | Part des cancers effectivement détectés |
| F1-score | 96,5 % | Équilibre entre précision et rappel |

**Matrice de confusion (jeu de test, 114 tumeurs) :**

| | Prédit bénin | Prédit malin |
|---|---|---|
| **Réellement bénin** | 70 | 2 (fausses alertes) |
| **Réellement malin** | 1 (cancer manqué) | 41 |

En contexte médical, le **rappel** est la métrique prioritaire : un cancer non détecté (faux négatif) est bien plus grave qu'une fausse alerte, qui sera écartée par des examens complémentaires. Ici, 41 cancers sur 42 sont détectés.

Le Perceptron **ne converge pas complètement** : le nombre d'erreurs par epoch ne descend jamais à 0. Les données ne sont donc pas parfaitement linéairement séparables, comme le laissait voir la zone de chevauchement de la projection ACP.

---

## 5. Outils utilisés

| Outil | Usage |
|---|---|
| Python 3 | Langage du projet |
| NumPy | Implémentation du Perceptron (calcul vectoriel) |
| pandas | Chargement et nettoyage des données |
| Matplotlib / Seaborn | Visualisations |
| scikit-learn | Séparation train/test, standardisation, ACP, métriques d'évaluation |
| Jupyter Notebook | Développement et présentation de l'analyse |
| Git / GitHub | Versionnement |

Le Perceptron est **entièrement codé à la main** : scikit-learn ne sert qu'à la préparation des données et à l'évaluation.

---

## 6. Structure du dépôt et installation

```
building-perceptron/
├── data/
│   └── data.csv          # Jeu de données Breast Cancer Wisconsin
├── notebook.ipynb        # Analyse complète : nettoyage, EDA, ACP, modélisation, évaluation
├── script.py             # Classe Perceptron
└── README.md
```

### Installation

```bash
git clone https://github.com/dina-yahiaoui/building-perceptron.git
cd building-perceptron
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn ipykernel
```

Ouvrir ensuite `notebook.ipynb` et exécuter toutes les cellules.

### Utilisation de la classe

```python
from script import Perceptron

model = Perceptron(learning_rate=0.01, n_epochs=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.score(X_test, y_test))
```

---

## 7. Conclusion

### Efficacité du Perceptron

Malgré sa simplicité, le Perceptron obtient de **très bons résultats** sur ce problème : 97,4 % d'accuracy et 97,6 % de rappel. Ce succès s'explique par la nature des données : après standardisation et ACP, les tumeurs bénignes et malignes sont presque linéairement séparables.

Il reste toutefois **insuffisant pour un usage médical réel** :

- **Absence de convergence** : les données ne sont pas parfaitement séparables, et la frontière finale dépend du nombre d'epochs et de l'ordre des exemples.
- **Pas de probabilité** : le modèle répond « bénin » ou « malin » sans niveau de confiance. Un médecin ne peut pas repérer les cas incertains à réexaminer.
- **Seuil non ajustable** : impossible de privilégier le rappel en abaissant le seuil de décision.
- **Frontière non optimale** : il s'arrête sur la première frontière acceptable, pas sur la meilleure.
- **Évaluation limitée** : un seul découpage train/test de 114 tumeurs, où une erreur fait varier le rappel de plus de 2 points.

### Solutions proposées (non implémentées)

| Solution | Apport |
|---|---|
| **Régression logistique** | Même frontière linéaire, mais fournit une probabilité et un seuil ajustable pour maximiser le rappel. Modèle interprétable. |
| **SVM (Support Vector Machine)** | Cherche la frontière qui maximise la marge entre les classes ; avec un noyau, gère les frontières non linéaires. |
| **Perceptron multicouche (MLP)** | Couches cachées et activations non linéaires (ReLU, sigmoïde), entraîné par rétropropagation. Lève la limite de linéarité. |
| **Random Forest / Gradient Boosting** | Très performants sur les données tabulaires et donnent l'importance de chaque variable. |
| **Validation croisée** | Évaluer sur plusieurs découpages pour des résultats plus fiables qu'un seul jeu de test. |
| **Algorithme de la poche (*pocket algorithm*)** | Variante du Perceptron qui conserve les meilleurs poids rencontrés quand les données ne sont pas séparables. |

---

## 8. Bibliographie

- Rosenblatt, F. (1958). [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://pubmed.ncbi.nlm.nih.gov/13602029/). *Psychological Review*, 65(6), 386-408.
- Minsky, M. & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.
- Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533-536.
- Street, W. N., Wolberg, W. H. & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE International Symposium on Electronic Imaging*, 1905, 861-870.
- Wolberg, W., Mangasarian, O., Street, N. & Street, W. (1995). [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). UCI Machine Learning Repository.
- Jumper, J. et al. (2021). [Highly accurate protein structure prediction with AlphaFold](https://pubmed.ncbi.nlm.nih.gov/34265844/). *Nature*, 596, 583-589.
- Lotter, W. et al. (2021). [Robust breast cancer detection in mammography and digital breast tomosynthesis using an annotation-efficient deep learning approach](https://deephealth.com/evidence/scientific-publications/robust-breast-cancer-detection-in-mammography-and-digital-breast-tomosynthesis-using-an-annotation-efficient-deep-learning-approach). *Nature Medicine*.
- Documentation [scikit-learn](https://scikit-learn.org/stable/) : `StandardScaler`, `PCA`, métriques de classification.
- Vidéos de la base de connaissances du projet : *Perceptron Algorithm with Code Example - ML for beginners!* et *Gradient Descent - Simply Explained! ML for beginners with Code Example!*

---

