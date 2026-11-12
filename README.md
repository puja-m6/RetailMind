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

