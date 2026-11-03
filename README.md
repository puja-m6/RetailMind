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

## Day 3 - Multi-Model Comparison, Tuning & Bias-Variance

This increment tunes and validates the competing churn models using `GridSearchCV` with 5-fold cross-validation. Training and validation recall are compared to assess generalisation, and learning-curve results are used to inspect how performance changes as training data increases.

### Hyperparameter Tuning

The following models were tuned:

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Churn recall was used as the GridSearchCV scoring metric so that model selection remains focused on identifying customers at risk of churn.

### Tuning Results

| Model | Best CV Recall | Train Recall | Train-Validation Gap | Best Hyperparameters |
|---|---:|---:|---:|---|
| Decision Tree | 0.5425 | 0.5564 | 0.0139 | `max_depth=5, min_samples_leaf=4, min_samples_split=2` |
| Random Forest | 0.5224 | 0.9120 | 0.3896 | `max_depth=12, min_samples_split=2, n_estimators=100` |
| SVM | 0.5338 | 0.5425 | 0.0087 | `C=10, gamma=scale, kernel=linear` |

### Bias-Variance Analysis

The training-versus-validation recall comparison shows different generalisation behaviour across the tuned models.

- **Decision Tree:** Train recall = 0.5564 and CV recall = 0.5425, producing a small gap of 0.0139.
- **Random Forest:** Train recall = 0.9120 and CV recall = 0.5224, producing a much larger gap of 0.3896. This result shows substantially stronger training performance than cross-validation performance and is evidence of high variance/overfitting relative to the other candidates.
- **SVM:** Train recall = 0.5425 and CV recall = 0.5338, producing the smallest gap of 0.0087.

### Tuned Champion

**Selected model:** Decision Tree

**Best cross-validation recall:** 0.5425

**Chosen hyperparameters:**

- `max_depth = 5`
- `min_samples_leaf = 4`
- `min_samples_split = 2`

Decision Tree was selected because it achieved the highest cross-validation recall among the three tuned candidates while maintaining a small train-validation gap.

The tuned champion has been versioned as:

`models/churn_v3.joblib`

### Connection to Earlier Increments

Day 3 builds on the same churn dataset and preprocessing approach used in the earlier RetailMind increments. The earlier model comparison is extended by systematically tuning the candidate classifiers with GridSearchCV and evaluating generalisation through cross-validation and train-versus-validation performance.

### Business Value

Hyperparameter tuning and cross-validation provide a more robust basis for selecting the churn model than relying on a single train-test result. The tuned Decision Tree improves the model-selection process while retaining the project's focus on churn recall and identifying at-risk customers.

### Open Blockers

No open blockers for the Day 3 increment.
