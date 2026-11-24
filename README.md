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

## Day 5 - Neural Networks & ANN (Keras)

This increment introduces a Keras feed-forward artificial neural network for customer churn and benchmarks it head-to-head against the versioned classical ML champion from the earlier RetailMind increments.

### ANN Architecture

- Normalised numerical inputs and one-hot encoded categorical inputs
- Dense hidden layer: 64 units, ReLU activation
- Dense hidden layer: 32 units, ReLU activation
- Output layer: 1 unit, sigmoid activation
- Optimizer: Adam
- Loss: Binary cross-entropy
- Early stopping: monitor validation loss, patience of 5 epochs, restore best weights

### ANN Training

- Epochs run: 8
- Best validation loss: 0.4277

### ANN vs Classical Champion

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Classical Champion | 0.7896 | 0.6021 | 0.6150 | 0.8289 |
| Keras ANN | 0.7946 | 0.6557 | 0.4786 | 0.8337 |

### When to Use Which

- **Classical champion:** Prefer when churn recall is the priority. It detected a substantially larger proportion of churn cases, with recall of 0.6150 compared with 0.4786 for the ANN.
- **Keras ANN:** The ANN produced slightly higher accuracy (0.7946 vs 0.7896), precision (0.6557 vs 0.6021), and ROC-AUC (0.8337 vs 0.8289). It is useful when those metrics and a neural-network architecture are more important than maximizing churn recall.
- **Day 5 conclusion:** The ANN did not clearly replace the classical champion for RetailMind's churn objective. Although the ANN marginally improved accuracy and ROC-AUC, the classical champion retained a major advantage in churn recall, which has been the project's primary business metric for identifying at-risk customers.

### Saved Artifacts

The trained ANN is saved as:

`models/churn_ann.keras`

The fitted preprocessing needed to reproduce ANN inputs is saved as:

`models/churn_ann_preprocessor.joblib`

### Connection to Earlier Increments

Day 5 keeps the existing RetailMind churn dataset and directly benchmarks deep learning against the previously versioned classical churn champion. This tests whether neural-network depth adds measurable value instead of automatically replacing the simpler classical approach.

### Business Value

The benchmark shows that model complexity should be chosen according to the business objective rather than accuracy alone. For churn intervention, the classical champion remains preferable because its recall is 0.1364 higher than the ANN's, while the ANN demonstrates modest gains in accuracy, precision, and ROC-AUC.

### Open Blockers

No open blockers for the Day 5 increment.

## Day 6 - CNN Image Classification

This increment adds RetailMind's first computer-vision module: a convolutional neural network that classifies product images into catalog categories.

### Prototype Dataset

The CNN was prototyped on Fashion-MNIST to validate the image-classification architecture before extending the vision workflow to real Fashion Product Images.

### Image Pipeline

- Normalised image pixel values
- Added lightweight image augmentation
- Used `Conv2D` layers for feature extraction
- Used `MaxPooling2D` for spatial reduction
- Used `Dropout` to reduce overfitting
- Used Dense layers for classification
- Used a 10-class softmax output

### CNN Results

- Epochs run: **15**
- Best validation accuracy: **0.8903**
- Test accuracy: **0.8854**
- Test loss: **0.3143**

### Confusion-Matrix Findings

The confusion matrix shows that the largest errors occur between visually similar clothing categories.

Top observed category confusions:

1. `T-shirt/top -> Shirt: 136`
2. `Coat -> Shirt: 104`
3. `Pullover -> Shirt: 101`
4. `Shirt -> T-shirt/top: 101`
5. `Coat -> Pullover: 82`

These results indicate that upper-body clothing categories such as shirts, T-shirts, coats, and pullovers are the most difficult for the current CNN to distinguish.

### Saved Artifact

The first RetailMind computer-vision classifier is saved as:

`models/catalog_cnn.keras`

### Connection to Earlier Increments

Earlier RetailMind increments focused on tabular customer-churn prediction. Day 6 adds a separate computer-vision capability for product images, extending the same RetailMind codebase toward automatic catalog tagging and future visual-search workflows.

### Business Value

The CNN reached **88.54% test accuracy**, demonstrating that RetailMind can automatically assign image-based product categories with useful prototype performance. Automated categorisation can reduce manual cataloguing effort, while the confusion analysis identifies visually similar categories that could benefit from more representative product images, stronger augmentation, or the transfer-learning work planned for the next increment.

### Open Blockers

No open blockers for the Day 6 increment.

## Day 7 - Transfer Learning & Object Detection

This increment upgrades RetailMind's computer-vision capability by introducing transfer learning with a pretrained MobileNetV2 model and object detection using YOLO.

### Transfer Learning with MobileNetV2

An ImageNet-pretrained MobileNetV2 backbone was used as a lightweight transfer-learning model suitable for the resource-constrained work-laptop environment.

The pretrained backbone was initially frozen while a new classification head was trained. The upper MobileNetV2 layers were then fine-tuned using a lower learning rate.

A reduced Fashion-MNIST subset and smaller image/batch sizes were used to keep the experiment practical on the available CPU-only environment.

### Classification Results

| Model | Test Accuracy |
|---|---:|
| Day 6 Custom CNN | 0.8854 |
| MobileNetV2 Transfer Learning | 0.8180 |

- Accuracy change: **-0.0674**
- MobileNetV2 test loss: **0.5175**

The Day 6 custom CNN outperformed MobileNetV2 by approximately **6.74 percentage points** on this experiment.

Transfer learning therefore did not improve classification accuracy in this specific setup. Possible contributing factors include the difference between ImageNet-style RGB images and the small grayscale Fashion-MNIST images, as well as the deliberately reduced training dataset used to accommodate the available hardware resources.

### Saved Transfer-Learning Model

The trained MobileNetV2 classifier was saved as:

`models/catalog_transfer_mobilenetv2.keras`

The transfer-learning metrics were saved as:

`models/day7_mobilenet_metrics.json`

### YOLO Object Detection

A pretrained YOLO detector was used to extend RetailMind from image-level classification to object localization.

The detector was tested on:

`data/product_images/sample.jpg`

### Detection Result

- Images processed: **1**
- Bounding boxes detected: **1**
- Detected class: **suitcase**
- Confidence: **0.532 (53.2%)**
- Bounding box: **[90.1, 204.3, 164.5, 272.1]**

YOLO successfully localized an object in the retail image and placed a bounding box around it.

However, several other visible fashion products were not detected. This highlights a limitation of using a general-purpose pretrained object detector for specialized retail and fashion imagery.

A future improvement could use a retail-specific object-detection dataset and fine-tune YOLO on classes such as bags, shoes, clothing, hats, and other catalog products.

### Saved Object-Detection Artifacts

Detection metadata was saved as:

`models/day7_yolo_results.json`

The annotated detection image was saved under:

`models/yolo_predictions/day7/`

### Resource-Constrained Execution

Running TensorFlow/MobileNetV2 and PyTorch/YOLO together caused the Jupyter kernel to restart because of the limited resources available on the work laptop.

To complete the increment reliably, the notebook was divided into two execution stages while remaining in the same notebook:

1. Train, evaluate, and save MobileNetV2.
2. Restart the Jupyter kernel to release TensorFlow resources.
3. Continue from the YOLO section without rerunning MobileNetV2.
4. Load the saved MobileNetV2 metrics when creating the final Day 7 summary.

This allowed both transfer learning and object detection to be completed without removing either requirement.

### Business Value

Day 7 extends RetailMind beyond basic image classification.

MobileNetV2 demonstrates how pretrained visual models can be adapted to RetailMind while using a relatively lightweight architecture.

YOLO adds object localization, enabling future capabilities such as:

- automatic catalog quality checks
- identifying products within multi-product images
- product counting
- shelf-image analysis
- locating products before further classification
- retail-specific visual search workflows

### Day 7 Conclusion

The Day 6 custom CNN remained the stronger classifier for the current Fashion-MNIST experiment, achieving **88.54% accuracy compared with 81.80% for MobileNetV2**.

MobileNetV2 nevertheless demonstrated the transfer-learning workflow with a substantially lighter architecture than ResNet50.

YOLO successfully added object detection and bounding-box localization to RetailMind. The prototype detected one object as a suitcase with **53.2% confidence**, demonstrating the detection pipeline while also showing the need for retail-specific fine-tuning for more complete product recognition.

### Open Blockers

The current pretrained YOLO detector is general-purpose and does not reliably detect all fashion products in the catalog image. Retail-specific YOLO fine-tuning would be required for more complete and accurate product detection.

## Day 8 - Segmentation & End-to-End Computer Vision

This increment completes Phase 2 of RetailMind by adding product segmentation, background removal, clean catalog-image compositing, Grad-CAM explainability, and a packaged end-to-end computer-vision interface.

### Input Image

The Day 8 workflow processes:

`data/product_images/sample.jpg`

### Product Segmentation

A lightweight segmentation approach was implemented using OpenCV GrabCut followed by morphological refinement.

The workflow:

1. Estimates foreground and background pixels with GrabCut.
2. Converts the GrabCut result into a binary foreground mask.
3. Applies morphological closing and opening to refine the mask.
4. Uses the final mask to remove the original background.

The foreground ratio was:

**0.424**

This means approximately **42.4%** of the image pixels were identified as foreground/product content.

### Background Removal and Catalog Compositing

The segmentation mask was used to create:

- a transparent-background product image
- a clean product image composited onto a uniform white background

The generated catalog artifacts are:

`models/day8_cv_outputs/product_mask.png`

`models/day8_cv_outputs/product_transparent.png`

`models/day8_cv_outputs/clean_catalog_white.jpg`

### Classification Result

The existing Day 7 MobileNetV2 transfer-learning classifier was reused for the Day 8 explainability workflow.

- Classifier: **MobileNetV2 transfer classifier**
- Predicted class: **Dress**
- Prediction score: **0.3699 (36.99%)**

The relatively low prediction score reflects the domain difference between the classifier's Fashion-MNIST training data and the real, multi-product retail image used for Day 8.

### Grad-CAM Explainability

Grad-CAM was added to show which image regions influenced the classifier's prediction.

The MobileNetV2 feature layer used for Grad-CAM was:

`out_relu`

The Grad-CAM result is saved as:

`models/day8_cv_outputs/gradcam_overlay.jpg`

This overlay provides visual evidence of the regions that contributed to the model's classification decision, helping the catalog team review the model rather than relying only on a class label.

### End-to-End CV Module

Day 8 packages the computer-vision capabilities into one workflow that includes:

- image classification
- Day 7 object-detection metadata
- product segmentation
- background removal
- uniform catalog-image compositing
- Grad-CAM explainability

The packaged module metadata is saved as:

`models/day8_cv_module.json`

The notebook exposes the combined result through the `retailmind_cv()` interface.

### Connection to Earlier Increments

Day 6 introduced the first CNN catalog classifier.

Day 7 added MobileNetV2 transfer learning and YOLO object detection.

Day 8 completes the Phase 2 computer-vision workflow by adding segmentation, clean-image generation, and explainability while reusing the earlier classifier and saved detection results.

### Business Value

This increment supports cleaner and more consistent retail catalog imagery by automatically separating product content from the original background and placing it on a uniform white background.

Grad-CAM adds transparency by showing which image regions influenced the classifier. This helps catalog teams review questionable predictions and builds trust in model-assisted catalog decisions.

The packaged CV workflow creates a foundation for:

- automated catalog-image cleanup
- consistent product presentation
- explainable classification review
- catalog quality assurance
- combined classification, detection, and segmentation workflows

### Limitations and Open Blockers

GrabCut is a lightweight segmentation prototype rather than a retail-trained semantic or instance-segmentation model. Complex backgrounds, overlapping objects, and multiple products may produce imperfect boundaries.

The MobileNetV2 classifier was trained using a reduced Fashion-MNIST experiment, while the Day 8 input is a real multi-product retail image. This domain shift contributed to the relatively low prediction score of **0.3699**.

A future production version should use:

- real fashion-product training images
- product-level segmentation masks
- a U-Net-style model or a pretrained semantic/instance segmenter
- retail-specific classifier and detector fine-tuning
- GPU-backed training for larger datasets and stronger models

### Day 8 Conclusion

Day 8 successfully produced a segmentation mask, transparent product image, clean white-background catalog image, and Grad-CAM explanation for the existing RetailMind classifier.

All required outputs were saved and verified, and classification, detection metadata, segmentation, compositing, and explainability were packaged into one computer-vision module.

## Day 9 - RNN / LSTM Demand Forecasting

This increment starts Phase 3 of RetailMind by adding a next-period demand-forecasting module for store and product sales.

### Dataset and Time Series

The model uses the **Store Item Demand** dataset with the columns:

- `date`
- `store`
- `item`
- `sales`

The Day 9 prototype forecasts the series:

`store_1.0_item_1.0`

The sales records were sorted by date and resampled into a clean daily demand series using pandas.

### Supervised Forecasting Windows

A **30-day look-back window** was used.

Each training example follows this structure:

`previous 30 days of sales -> next-day sales`

The series was scaled using statistics calculated only from the training period.

### Time-Series Split

The split was chronological rather than random so that the model learned from earlier observations and was evaluated on future demand.

- Training observations: **1460**
- Test observations: **366**

The most recent observations were held out for testing.

### LSTM Architecture

The forecasting model contains:

- LSTM layer with 32 units
- Dropout layer with rate 0.20
- Dense hidden layer with 16 ReLU units
- Final Dense layer with one output for next-period demand
- Adam optimizer
- Mean squared error training loss
- Early stopping using validation loss

### Training Result

- Epochs run: **20**
- Look-back period: **30 days**

### Forecast Evaluation

| Metric | Result |
|---|---:|
| MAE | 4.5853 |
| RMSE | 5.8225 |

The MAE indicates that the forecast differed from actual demand by approximately **4.59 sales units on average**.

The RMSE of **5.8225** gives greater weight to larger forecast errors and provides an additional measure of forecast reliability.

### Next-Period Forecast

Using the latest 30-day demand window, the model predicted next-period demand of:

**17.57 units**

This value can be used as an initial demand signal for inventory and replenishment planning.

### Saved Artifacts

The trained LSTM forecaster was saved as:

`models/demand_lstm.keras`

The forecast metrics were saved as:

`models/day9_demand_metrics.json`

The cleaned daily demand series was saved as:

`models/day9_forecast_outputs/daily_demand_series.csv`

The actual-versus-predicted test output was saved as:

`models/day9_forecast_outputs/demand_forecast.csv`

### Demand Module Interface

The notebook provides the function:

`forecast_next_period(recent_sales)`

The function accepts exactly 30 recent daily sales values and returns the next-period sales forecast.

### Connection to Earlier Increments

Days 6 to 8 developed RetailMind's computer-vision catalog capabilities.

Day 9 begins Phase 3 by adding time-series demand forecasting to the same RetailMind platform. This expands RetailMind from customer and catalog intelligence into inventory-planning support.

### Business Value

The LSTM demand module provides a forward-looking sales signal that can support:

- inventory reorder decisions
- stockout reduction
- overstock reduction
- store and product planning
- replenishment prioritisation
- future safety-stock calculations

A forecast of **17.57 units** for the next period gives inventory teams a measurable starting point for operational planning.

### Limitations and Open Blockers

The Day 9 prototype forecasts only one store/item series and performs one-step-ahead prediction.

The current model does not yet include external factors such as:

- promotions
- pricing
- holidays
- weather
- store closures
- product availability

The next increment can extend the module to multi-horizon forecasting using a Seq2Seq model with attention and can support multiple store/product series.

### Day 9 Conclusion

Day 9 successfully converted store/item sales into a daily time series, generated 30-day supervised windows, trained an LSTM, evaluated future-period predictions with MAE and RMSE, and saved the model and forecast outputs.

The final model achieved an MAE of **4.5853**, an RMSE of **5.8225**, and predicted next-period demand of **17.57 units**.

## Day 10 - Seq2Seq & Attention Forecasting

This increment upgrades RetailMind's Day 9 single-step LSTM demand forecaster into a multi-horizon Seq2Seq model with attention.

### Forecasting Objective

Day 9 predicted one next-period demand value from the previous 30 days. Day 10 extends the demand module to predict the next **7 days** in one forecast while using attention to identify which historical demand periods influence each future step.

### Model Design

The model uses:

- 30-day historical look-back window
- LSTM encoder
- LSTM decoder
- Keras AdditiveAttention over encoder history
- 7-day multi-horizon output
- Adam optimizer
- Mean squared error training loss
- Early stopping using validation loss

### Training Result

- Look-back window: **30 days**
- Forecast horizon: **7 days**
- Epochs run: **18**

### Multi-Horizon Forecast Evaluation

| Metric | Result |
|---|---:|
| Overall 7-day MAE | 4.5754 |
| Overall 7-day RMSE | 5.8059 |
| Day +1 MAE | 4.5988 |
| Day +1 RMSE | 5.8311 |

The overall MAE indicates an average absolute forecast error of approximately **4.58 sales units** across the seven forecast horizons.

### Comparison with Day 9

| Metric | Day 9 Single-Step LSTM | Day 10 Seq2Seq Day +1 |
|---|---:|---:|
| MAE | 4.5853 | 4.5988 |
| RMSE | 5.8225 | 5.8311 |

The Day 10 Day +1 forecast is slightly less accurate than the dedicated Day 9 single-step model. However, Day 10 adds substantially more functionality by producing seven future demand values simultaneously and exposing attention weights for interpretability.

The improvement in Day 10 is therefore primarily **forecast horizon and explainability**, rather than a one-step accuracy improvement.

### Next 7-Day Demand Forecast

The model produced the following demand outlook:

| Forecast Horizon | Predicted Demand |
|---|---:|
| Day +1 | 16.86 |
| Day +2 | 17.05 |
| Day +3 | 17.28 |
| Day +4 | 17.51 |
| Day +5 | 17.74 |
| Day +6 | 17.95 |
| Day +7 | 18.13 |

The forecast shows a gradual increase in expected demand from approximately **16.86 units to 18.13 units** over the seven-day horizon.

### Attention Explainability

The Seq2Seq decoder uses attention over the previous 30 days of demand history.

For each forecast horizon, the notebook ranks the historical lags receiving the highest attention weights. This provides evidence of which past demand periods contribute most strongly to each future prediction.

The saved attention output is:

`models/day10_forecast_outputs/attention_weights.csv`

This file can be used to inspect the highest-weighted historical periods for Day +1 through Day +7 forecasts.

### Upgraded Demand Interface

The Day 10 notebook provides:

`forecast_multi_horizon(recent_sales)`

The function accepts exactly 30 recent daily sales values and returns seven future demand predictions.

This upgrades the Day 9 interface from:

`30 historical days -> 1 future day`

to:

`30 historical days -> 7 future days`

### Saved Artifacts

The trained Seq2Seq + attention model was saved as:

`models/demand_seq2seq_attention.keras`

The Day 10 metrics were saved as:

`models/day10_demand_metrics.json`

The multi-horizon forecast results were saved as:

`models/day10_forecast_outputs/multi_horizon_forecast.csv`

The attention weights were saved as:

`models/day10_forecast_outputs/attention_weights.csv`

### Connection to Day 9

Day 9 established RetailMind's initial next-period LSTM demand model using a 30-day look-back window.

Day 10 reuses the persisted Day 9 daily demand series and expands the forecasting module into a Seq2Seq architecture capable of predicting multiple future periods. Attention adds interpretability by showing which historical demand lags receive the most weight for each forecast horizon.

### Business Value

The seven-day demand outlook provides inventory and procurement teams with more planning lead time than a one-day forecast.

The module can support:

- multi-day replenishment planning
- procurement timing
- promotion planning
- inventory allocation
- stockout prevention
- overstock reduction
- identification of rising or falling short-term demand patterns

The attention mechanism also improves model transparency by showing which recent demand periods contribute to each prediction.

### Limitations and Open Blockers

The current Phase 3 prototype still models a single store/item demand series.

The model currently uses historical sales only and does not yet include external demand drivers such as:

- promotions
- pricing
- holidays
- weather
- store closures
- product availability
- regional events

The Day +1 Seq2Seq forecast was also slightly less accurate than the Day 9 dedicated single-step LSTM. Future tuning should therefore evaluate architecture size, look-back length, forecast horizon, attention configuration, and additional demand features rather than assuming that a more complex architecture always improves accuracy.

### Day 10 Conclusion

Day 10 successfully upgraded RetailMind from single-step demand forecasting to a seven-day Seq2Seq forecasting module with attention.

The model achieved an overall MAE of **4.5754** and RMSE of **5.8059** across the seven-day horizon. It generated a demand outlook rising from **16.86 units on Day +1 to 18.13 units on Day +7** and persisted attention weights that explain which historical demand periods influenced each forecast.

## Day 11 - GANs & VAEs: Image Augmentation

This increment uses a Variational Autoencoder (VAE) to generate synthetic product images for a category the classifier struggles with, adds them to the training data, and re-measures per-class accuracy.

### Weak Category

The Day 6 confusion matrix showed that **Shirt** was confused most often with T-shirt/top, Pullover and Coat, so Shirt was chosen as the target category.

Fashion-MNIST has the same number of images for every class, so Shirt was deliberately made a thin category:

- Shirt: **600** real images
- Every other class: up to **3000** real images

### VAE

A small VAE (encoder and decoder) was trained on the 600 real Shirt images only, with a 16-number latent space, for 50 epochs.

After training, **1000** random latent vectors were passed through the decoder to generate new Shirt images.

### Quality Check

The generated images were inspected visually. Only realistic images were kept: an image was kept if its average brightness fell between the 5th and 95th percentile of the real Shirt images. The notebook kept **300 of the 1000** generated images (the notebook caps the kept set at 300) and added them to the training set, labelled as Shirt.

The real, generated and kept images are saved in `models/day11_augmentation_outputs/`.

### Per-Class Accuracy Before and After

The same CNN architecture and the same starting seed were used both times, so the only difference is the added synthetic Shirt images.

| Class | Before | After | Change |
|---|---:|---:|---:|
| T-shirt/top | 0.9530 | 0.9390 | -0.0140 |
| Trouser | 0.9600 | 0.9750 | +0.0150 |
| Pullover | 0.7610 | 0.8250 | +0.0640 |
| Dress | 0.8640 | 0.9130 | +0.0490 |
| Coat | 0.9220 | 0.8970 | -0.0250 |
| Sandal | 0.9700 | 0.9810 | +0.0110 |
| **Shirt (target)** | **0.1670** | **0.2180** | **+0.0510** |
| Sneaker | 0.9710 | 0.9570 | -0.0140 |
| Bag | 0.9820 | 0.9550 | -0.0270 |
| Ankle boot | 0.9570 | 0.9430 | -0.0140 |

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Shirt accuracy | 0.1670 | 0.2180 | +0.0510 |
| Overall accuracy | 0.8507 | 0.8603 | +0.0096 |

### Fairness and Accuracy Lift

Shirt accuracy improved from **16.7% to 21.8%** after augmentation, and overall accuracy rose slightly from **85.07% to 86.03%** (+0.0096).

Pullover (+0.0640), Dress (+0.0490), Trouser (+0.0150) and Sandal (+0.0110) also improved. T-shirt/top, Coat, Sneaker, Bag and Ankle boot each dropped by between 0.0140 and 0.0270.

Shirt accuracy is still very low, so augmentation reduced the gap but did not fix the weak category. These numbers come from a single training run, so small differences of a few percentage points may partly be random variation.

### Saved Artifacts

- `models/day11_vae_decoder.keras` - the trained VAE decoder used to generate images
- `models/day11_augmented_classifier.keras` - the classifier trained with the synthetic Shirt images
- `models/day11_augmentation_metrics.json` - the before/after results
- `models/day11_augmentation_outputs/real_shirts.png`
- `models/day11_augmentation_outputs/generated_shirts.png`
- `models/day11_augmentation_outputs/kept_shirts.png`

### Connection to Earlier Increments

Day 6 built the CNN catalog classifier and produced the confusion matrix that identified Shirt as the weak category. Day 11 uses that finding to generate extra training images and tests whether they improve the same kind of classifier.

### Business Value

Synthetic images can strengthen thin product categories without collecting and labelling new photos. Measuring accuracy per class before and after shows whether the extra data actually helps each category, instead of assuming it does.

### Open Blockers

- The experiment uses 28x28 Fashion-MNIST images, so VAE output is blurry. A real version should use the Kaggle Fashion Product Images dataset.
- The realism check only compares average brightness, which is a basic filter. Stronger checks or human review would be needed before synthetic images enter a real catalog.
- Results are from one run; repeating with different seeds would show how stable the improvement is.

## Day 12 - NLP & Classical Sentiment Baseline

This increment adds RetailMind's first natural-language processing module for converting Amazon product reviews into sentiment signals and customer-feedback insights.

### Dataset

The sentiment baseline used a balanced subset of the Amazon Polarity review dataset containing:

- **20,000 Amazon product reviews**
- **10,000 negative reviews**
- **10,000 positive reviews**

### Text Processing Pipeline

Each review passed through the following preprocessing steps:

1. Convert text to lowercase.
2. Remove HTML and non-alphabetic noise.
3. Tokenize the review using NLTK.
4. Remove English stopwords.
5. Convert the cleaned reviews into TF-IDF features.

The TF-IDF representation included both single words and two-word phrases.

### Classical Sentiment Model

A **Logistic Regression** classifier was trained on the TF-IDF review features to provide a transparent classical sentiment baseline.

### Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | **0.8718** |
| Precision | **0.8668** |
| Recall | **0.8785** |
| F1-score | **0.8726** |

The model achieved **87.18% accuracy** on the held-out Amazon review test set, with precision, recall, and F1-score all close to 87%.

### Sentiment Confusion Matrix

The notebook includes a confusion-matrix visual for negative and positive reviews. This provides a direct view of correct classifications and sentiment errors rather than relying only on a single accuracy score.

### Informative Positive Terms

The strongest positive TF-IDF terms identified by the Logistic Regression model included:

- `great`
- `excellent`
- `love`
- `best`
- `perfect`
- `easy`
- `awesome`
- `favorite`

These terms provide interpretable examples of language associated with positive customer sentiment.

### Informative Negative Terms

The strongest negative terms included:

- `boring`
- `disappointed`
- `worst`
- `disappointing`
- `waste`
- `terrible`
- `poor`
- `money`

These terms can help RetailMind surface recurring dissatisfaction themes and product-quality concerns from review text.

### Reusable Sentiment Interface

The notebook provides:

`predict_sentiment(review)`

The function accepts a new review and returns:

- predicted sentiment: positive or negative
- prediction confidence

This packages the classical NLP pipeline into a reusable review-processing interface for later RetailMind increments.

### Saved Artifacts

The fitted Logistic Regression sentiment classifier is saved as:

`models/day12_sentiment_logreg.pkl`

The fitted TF-IDF vectorizer is saved as:

`models/day12_tfidf.pkl`

The evaluation metrics are saved as:

`models/day12_nlp_metrics.json`

The most informative sentiment terms are saved as:

`models/day12_nlp_outputs/informative_terms.json`

### Connection to Earlier Increments

Earlier RetailMind phases added customer churn prediction, computer vision, demand forecasting, and synthetic image augmentation.

Day 12 begins the NLP capability by converting free-text customer reviews into structured positive/negative sentiment signals and interpretable product-feedback terms.

### Business Value

The sentiment module provides a scalable way to convert large volumes of customer review text into measurable feedback signals.

RetailMind can use these signals to support:

- customer-satisfaction monitoring
- product issue discovery
- identification of recurring positive product attributes
- review prioritisation
- catalog and merchandising decisions
- future sentiment dashboards and trend analysis

The informative terms also give business users a simple explanation of which words are most strongly associated with customer satisfaction or dissatisfaction.

### Limitations and Open Blockers

The TF-IDF + Logistic Regression baseline does not deeply understand context, word order, sarcasm, or nuanced language.

For example, phrases with negation or complex context may be harder for a bag-of-words-style representation to understand correctly.

A future transformer/BERT increment can be compared directly against this classical baseline to determine whether richer contextual language modeling produces a meaningful improvement.

### Day 12 Conclusion

Day 12 successfully created RetailMind's first classical NLP sentiment module using Amazon product reviews, NLTK preprocessing, TF-IDF features, and Logistic Regression.

The model achieved **87.18% accuracy**, **86.68% precision**, **87.85% recall**, and an **87.26% F1-score**. The pipeline also surfaced interpretable positive and negative review terms and persisted the trained model and vectorizer for reuse in later NLP increments.

