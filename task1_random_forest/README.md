# Random Forest Classification — From Scratch

Titanic survival prediction using a Random Forest implemented entirely from scratch (`DecisionTree` + `RandomForest` classes, no `sklearn.ensemble`/`sklearn.tree`) — see `Random_Forest_From_Scratch.ipynb`.

## Approach

- Data: the classic 891-row labeled Titanic set (same columns/content as Kaggle's `train.csv`), loaded directly by URL — see the notebook's intro cell for why (Kaggle's official `test.csv` has no labels to score against).
- Preprocessing: median-by-title age imputation, mode Embarked imputation, `HasCabin` flag, `Title`/`FamilySize`/`IsAlone` engineered features, binned Age/Fare, encoded categoricals.
- Model: `DecisionTree` (Gini impurity, recursive best-split search) wrapped by `RandomForest` (bootstrap sampling + per-split random feature selection + majority vote).
- Manual hyperparameter sweeps over `n_estimators`, `max_depth`, `min_samples_split`.
- Accuracy, confusion matrix, precision/recall/F1, and feature importance are all computed with hand-written functions, not `sklearn.metrics`.

## How to run

Open `Random_Forest_From_Scratch.ipynb` in Colab or Jupyter and run all cells top to bottom — no GPU or external dataset download needed beyond the one public CSV URL.

## Report

See Part 5 in the notebook for the write-up (results, feature importance, limitations) — fill in the numbers after a run, then this section doubles as the 1-2 page summary report deliverable.

## Colab link

_Paste the Colab link here after uploading/running._
