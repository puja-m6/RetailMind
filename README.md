# RetailMind AI

RetailMind is an incremental retail AI platform.

## Day 1 - Customer Churn Prediction

This increment establishes the initial machine-learning pipeline for identifying customers at risk of churn using the Telco Customer Churn dataset.

## Project structure

- `data/` - dataset
- `notebooks/` - exploration notebook
- `src/` - reusable training and prediction code
- `models/` - persisted trained pipeline

## Day 1 pipeline

1. Load the Telco Customer Churn dataset with pandas.
2. Inspect and clean the data.
3. Split features into training and test sets.
4. Scale numerical features with `StandardScaler`.
5. Encode categorical features with `OneHotEncoder`.
6. Train a `LogisticRegression` baseline using a scikit-learn `Pipeline`.
7. Evaluate accuracy, precision, recall, and ROC-AUC.
8. Save the fitted pipeline with joblib.
9. Load the saved pipeline through a `predict()` helper.

## Baseline metrics

Run `python src/train.py` from the `RetailMind` root folder, then replace the placeholders below with the printed values.

- Accuracy:  0.8038379530916845
- Precision: 0.6484848484848484
- Recall: 0.5721925133689839
- ROC-AUC: 0.8359290473207676

## Day 2 - Decision Tree, Random Forest & SVM

This increment upgrades the churn baseline by comparing three classification models: Decision Tree, Random Forest, and Support Vector Machine (SVM).

The same preprocessing approach and train-test split were reused to ensure a fair comparison between the models.

### Model Comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Decision Tree | 0.7306 | 0.4934 | 0.4973 | 0.6565 |
| Random Forest | 0.7875 | 0.6307 | 0.4840 | 0.8137 |
| SVM | 0.7918 | 0.6401 | 0.4947 | 0.7886 |

### Champion Model

**Decision Tree** was selected as the Day 2 champion because it achieved the highest recall for the churn class at **0.4973**.

Recall was given priority because the objective of this increment is to identify as many at-risk customers as possible. Although SVM achieved the highest accuracy and precision, and Random Forest achieved the highest ROC-AUC, Decision Tree produced the highest churn recall.

The selected model has been persisted as:

`models/champion_model.joblib`