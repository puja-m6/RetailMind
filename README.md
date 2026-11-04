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

# Day 4 - Feature Engineering and Ensemble Models

## Objective

The objective of Day 4 was to improve the RetailMind customer churn prediction pipeline using feature engineering and ensemble learning techniques.

## Feature Engineering

New domain-based features were created from the existing Telco Customer Churn dataset:

- Tenure buckets
- New customer flag
- Long-term customer flag
- Average charge per tenure
- Monthly-to-total charge ratio
- Number of active services
- Protection service flag
- Month-to-month contract flag
- Long-term contract flag
- Electronic check payment flag
- Paperless billing flag

These engineered features provide additional information that may help the models identify customers with a higher likelihood of churn.

## Ensemble Models

Two ensemble models were trained and evaluated:

1. Random Forest Classifier
2. XGBoost Classifier

Random Forest represents a bagging-based ensemble approach, while XGBoost represents a boosting-based ensemble approach.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7807 | 0.5827 | 0.6123 | 0.5971 | 0.8339 |
| XGBoost | 0.7913 | 0.6342 | 0.5053 | 0.5625 | 0.8428 |

## Final Model Selection

**XGBoost was selected as the Day 4 champion model.**

XGBoost achieved the highest ROC-AUC score of **0.8428**, along with the highest accuracy and precision among the Day 4 models.

Random Forest achieved a higher recall of **0.6123** and a higher F1 score of **0.5971**, showing that the choice of model can depend on the business objective. For this implementation, ROC-AUC was used as the primary model-selection metric.

## Saved Model

The complete feature-engineering, preprocessing, and trained champion pipeline was saved as:

`models/churn_champion_day4.joblib`

The Day 4 model metrics and champion information were saved as:

`models/day4_metrics.json`

The saved model was successfully reloaded and tested to confirm that it can generate both churn predictions and churn probabilities.

## Business Value

The Day 4 pipeline improves churn-risk modelling by combining customer tenure, contract behaviour, service usage, payment behaviour, and charge-related information.

Churn probabilities can be used to rank customers by predicted churn risk. This can support targeted retention campaigns where higher-risk customers are prioritised for intervention.

## Day 4 Project Updates

New files added:

- `notebooks/day4_feature_engineering_ensembles.ipynb`
- `src/features.py`
- `src/train_day4.py`

Generated model artifacts:

- `models/churn_champion_day4.joblib`
- `models/day4_metrics.json`

## Day 4 Summary

- Implemented domain-based feature engineering
- Added reusable feature-engineering code
- Trained a Random Forest bagging model
- Trained an XGBoost boosting model
- Compared models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC
- Selected XGBoost as the Day 4 champion based primarily on ROC-AUC
- Saved the complete production pipeline
- Verified that the saved model can be reloaded and used for churn prediction

