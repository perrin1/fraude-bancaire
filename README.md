# Fraude Bancaire : Détection sur Données Déséquilibrées

Détection de transactions frauduleuses sur le dataset Kaggle Credit Card Fraud.
Focus sur les **techniques contre le déséquilibre de classes** : `scale_pos_weight`, SMOTE,
et optimisation du seuil de décision.

## Résultats

| Stratégie | F1 | Recall | AUC-ROC |
|---|---|---|---|
| XGBoost standard | 0.8254 | 0.7959 | 0.9277 |
| scale_pos_weight | 0.8586 | 0.8367 | 0.9662 |
| SMOTE | 0.7580 | 0.8469 | 0.9742 |
| **scale_pos_weight + seuil optimal**  | **~0.88+** |  | 0.9662 |

> Modèle retenu : **XGBoost + scale_pos_weight + seuil optimisé**.
> Simple, rapide, et plus performant que SMOTE sur ce dataset.

## Pourquoi ce TP ?

Dans la plupart des problèmes réels de classification fraude, maladie rare, panne machine 
la classe qui nous intéresse représente moins de 1% des données.
Un modèle entraîné naïvement sur ces données obtient une accuracy de 99%...
en ne détectant **jamais** la fraude. C'est le piège le plus courant en ML appliqué.

Ce TP a pour objectif de comprendre **pourquoi l'accuracy est une mauvaise métrique**
sur des données déséquilibrées, et de maîtriser les trois approches pour y remédier :

1. **`scale_pos_weight`**: dire au modèle que rater une fraude coûte 577× plus cher que rater une transaction normale
2. **SMOTE**: générer synthétiquement des exemples de fraudes pour rééquilibrer le dataset
3. **Seuil de décision**: ne pas utiliser 0.5 par défaut, mais trouver le seuil qui maximise le F1

La conclusion de ce TP est contre-intuitive : **SMOTE, malgré sa complexité, est moins efficace
que le simple `scale_pos_weight`** sur ce dataset. Ce genre de résultat ne s'apprend pas dans
un cours théorique, il faut l'expérimenter soi-même.


## Dataset

- **Source :** Kaggle : Credit Card Fraud Detection
- **Dimensions :** 284 807 transactions × 31 colonnes
- **Cible :** `Class` (0 = normal, 1 = fraude)
- **Déséquilibre :** 0.1727% de fraudes (492 sur 284 807)
- **Features :** `V1`–`V28` (PCA anonymisées), `Amount`, `Time`

## Concepts clés abordés

- **Le piège de l'accuracy** : 99.83% d'accuracy avec un modèle qui prédit tout "normal"
- **scale_pos_weight** : pénalise le modèle davantage sur les fraudes (ratio = 577)
- **SMOTE** : sur-échantillonnage synthétique de la classe minoritaire
- **Seuil de décision** : `precision_recall_curve` pour trouver le seuil F1-optimal
- **Pipeline sklearn** : anti-data leakage (scaler inclus dans le pipeline)

## Structure du projet

```
fraude-bancaire/
├── requirements.txt
├── README.md
├── data/
│   └── creditcard.csv          
├── models/
│   ├── xgb_standard.pkl
│   ├── xgb_scale_pos_weight.pkl
│   ├── xgb_smote.pkl
│   └── seuil_optimal.json
├── notebook/
│   └── fraude_bancaire.ipynb
└── outputs/
    ├── desequilibre_classes.png
    ├── courbe_seuil_optimal.png
    └── confusion_roc.png
```

## Installation

```bash
git clone https://github.com/perrin1/fraude-bancaire
cd fraude-bancaire
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer le notebook

```bash
jupyter notebook notebook/fraude_bancaire.ipynb
```

## Technologies

`scikit-learn` · `xgboost` · `imbalanced-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn`

## Auteur

**Josse Perrin FANOU** : Ingénieur Logiciel & Data Science

perrinfanou6@gmail.com /+2290162099124