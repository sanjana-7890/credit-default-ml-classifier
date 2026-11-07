# Credit Default Classification

## Problem Statement

Train and evaluate five machine-learning classifiers to predict whether a credit-card client will default on a payment next month. The Streamlit app evaluates one selected saved model at a time against labeled test data.

## Dataset Description

The project uses the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), containing 30,000 records, 23 predictor features, and a binary target. The original `ID` field is excluded from training. The target is `default_payment_next_month` (`1` = default, `0` = no default).

The original UCI workbook and its CSV copy are stored in `data/raw/`. The training script reads `data/raw/credit_default.csv`. A stratified 80/20 split with random state 42 creates the 6,000-row `test_data.csv`. The dataset is licensed under CC BY 4.0. Citation: Yeh, I. (2009), *Default of Credit Card Clients*, UCI Machine Learning Repository, https://doi.org/10.24432/C55S3H.

## GitHub Repository Link

[credit-default-ml-classification](https://github.com/sanjana-7890/credit-default-ml-classification)

## Models Used

All models use the same stratified 80/20 train/test split. A three-fold stratified grid search, optimizing F1, selects hyperparameters using only the training portion. The 6,000-row test set is held out from tuning. AUC is calculated from positive-class probabilities when available; the Linear SVM uses its decision-function scores. The measured scores below are from the fixed holdout split.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6797 | 0.7081 | 0.3672 | 0.6202 | 0.4613 | 0.2718 |
| Decision Tree | 0.7660 | 0.7465 | 0.4755 | 0.5622 | 0.5152 | 0.3646 |
| K-Nearest Neighbors | 0.8090 | 0.7354 | 0.6262 | 0.3384 | 0.4393 | 0.3595 |
| Gaussian Naive Bayes | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Support Vector Machine | 0.6852 | 0.7072 | 0.3723 | 0.6172 | 0.4644 | 0.2770 |

### Model Performance Observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Class weighting raises recall to 0.6202 and F1 to 0.4613, at the cost of accuracy (0.6797) and precision (0.3672). |
| Decision Tree | Highest AUC (0.7465), F1 (0.5152), and MCC (0.3646); it gives the strongest balance on the selected F1 tuning objective. |
| K-Nearest Neighbors | Highest accuracy (0.8090) and precision (0.6262), but recall is lower at 0.3384. |
| Gaussian Naive Bayes | Recall remains relatively high at 0.5539; F1 is 0.4975, below the tuned Decision Tree. |
| Support Vector Machine | Class weighting raises recall to 0.6172 and F1 to 0.4644, while accuracy is 0.6852. |
| Overall winner | Decision Tree is the winner when prioritizing F1, AUC, and MCC. K-Nearest Neighbors has the highest accuracy, so the preferred model depends on the metric that matters most. |

These observations describe this fixed split after tuning. Review the `model_tuning.json` choices and explain the trade-offs in your own words before submission.

## Run Locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The training script performs the cross-validated search, saves five fitted models under `model/`, writes scores to `model_metrics.csv`, stores selected parameters and CV F1 scores in `model_tuning.json`, and creates the labeled holdout file. The dark-themed app starts with dataset statistics, a sample preview, and target distribution; it offers the labeled holdout CSV for download. Evaluation starts with uploading a labeled CSV. After upload, the app confirms the row count and asks for a model selection; only then does it show that model's metrics and confusion matrix, with a downloadable PDF report. Uploaded CSVs must contain all 23 input features and the target; the app loads the saved model without retraining.

## Live Streamlit App

[Open the live Credit Default Classification app](https://credit-default-ml-classification-nw2ybnuvxjracbnr82aicc.streamlit.app/)