# Random Forest Classification (From Scratch)

Titanic survival prediction using a Random Forest implemented entirely from scratch (`DecisionTree` + `RandomForest` classes, no `sklearn.ensemble`/`sklearn.tree`). See `Random_Forest_From_Scratch.ipynb`.

## Approach

- Data: the classic 891-row labeled Titanic set (same columns/content as Kaggle's `train.csv`), loaded directly by URL. See the notebook's intro cell for why (Kaggle's official `test.csv` has no labels to score against).
- Preprocessing: median-by-title age imputation, mode Embarked imputation, `HasCabin` flag, `Title`/`FamilySize`/`IsAlone` engineered features, binned Age/Fare, encoded categoricals.
- Model: `DecisionTree` (Gini impurity, recursive best-split search) wrapped by `RandomForest` (bootstrap sampling + per-split random feature selection + majority vote).
- Manual hyperparameter sweeps over `n_estimators`, `max_depth`, `min_samples_split`.
- Accuracy, confusion matrix, precision/recall/F1, and feature importance are all computed with hand-written functions, not `sklearn.metrics`.

## How to run

Open `Random_Forest_From_Scratch.ipynb` in Colab or Jupyter and run all cells top to bottom. No GPU or external dataset download needed beyond the one public CSV URL.

## Report

Hyperparameter sweep (validation accuracy):
- `n_estimators`: 10 → 0.8146, 50 → 0.8202, 100 → **0.8258**, 200 → 0.8202 (diminishing/slightly negative returns past 100 trees)
- `max_depth`: 3 → 0.8034, 5 → 0.8202, 10 → **0.8315**, None → 0.8315 (no overfitting penalty at unlimited depth on this dataset size)
- `min_samples_split`: 2 → 0.8315, 5 → 0.8315, 10 → 0.8202

Final model (100 trees, unlimited depth) on the held-out validation split:
- **Accuracy: 0.8315, Precision: 0.8308, Recall: 0.7397, F1: 0.7826**
- Confusion matrix: 94 true negatives, 11 false positives, 19 false negatives, 54 true positives. The model misses more actual survivors than it wrongly flags, consistent with the lower recall vs. precision.

Feature importance (top 5): `Sex` (0.316) dominates by a wide margin, then `Title` (0.161), `Pclass` (0.100), `FareBin` (0.091), `FamilySize` (0.079), matching the historical "women and children first, and class mattered" account of the evacuation.

See Part 5 in the notebook for the fuller write-up (limitations, business insights).

## Colab link

Run locally (Python 3.12, no GPU needed) rather than in Colab; the notebook in this repo already has all cells executed with real outputs/plots. To reproduce: open in Jupyter/Colab and run all cells top to bottom, no external dataset download needed beyond the one public CSV URL.
