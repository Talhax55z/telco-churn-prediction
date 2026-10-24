# Telco Customer Churn: EDA & Model Comparison

Which customers leave, why, and how well can we flag them early? Two notebooks on the IBM Telco Customer Churn dataset: an EDA that finds the drivers of churn, and a modeling notebook that compares six classifiers with a setup that avoids the common resampling leak.

| Notebook | What's in it |
|---|---|
| [`Churn_Analysis_Enhanced_EDA.ipynb`](Churn_Analysis_Enhanced_EDA.ipynb) | Cleaning, churn rate by tenure/contract/payment/services, correlations, exports `telco_churn_processed.csv` |
| [`Churn_Analysis_Full_Improved_Model_Building.ipynb`](Churn_Analysis_Full_Improved_Model_Building.ipynb) | Stratified split, SMOTEENN, 4 base models + tuned random forest + voting ensemble, ROC/PR/calibration/learning curves, threshold analysis |

## What drives churn (EDA)

| Segment | Churn rate |
|---|:---:|
| Tenure under 12 months | ~47% |
| Electronic check payment | ~45% |
| Month-to-month contract | ~43% |
| No online security / no tech support | ~42% |
| Fiber optic internet | ~42% |
| Two-year contract | ~3% |

Churners pay more per month (median ~$75 vs ~$60) but have lower total charges because they leave early. Retention effort should focus on the first 12 months and on month-to-month customers.

## Modeling setup

- **Data:** 7,043 customers → 7,032 after dropping 11 rows with a blank `TotalCharges`. Churn rate 26.6%.
- **Split first, resample second:** stratified 80/20 split (5,625 train / 1,407 test, 374 churners in test). SMOTEENN is applied to the **training split only** (5,625 → 4,665 rows), and the scaler is fit on training data only, so no test information leaks into training.
- **Models:** logistic regression, decision tree, random forest, gradient boosting, a `RandomizedSearchCV`-tuned random forest, and a soft-voting ensemble.

### Results on the held-out test set

| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic regression | 0.763 | 0.541 | 0.698 | 0.610 | 0.826 |
| Decision tree | 0.748 | 0.519 | 0.709 | 0.599 | 0.798 |
| Random forest | 0.734 | 0.500 | 0.762 | 0.604 | 0.830 |
| Gradient boosting | 0.763 | 0.541 | 0.717 | 0.617 | 0.830 |
| **Tuned random forest** | 0.763 | 0.540 | 0.735 | **0.623** | 0.829 |
| Voting ensemble | 0.763 | 0.540 | 0.722 | 0.618 | **0.834** |

Accuracy is a poor yardstick here: predicting "no churn" for everyone scores about 73% and catches nobody, so the notebook selects on churn-class F1 and recall. The six models land within about 0.02 F1 of each other on 374 positive cases, so the differences between them are small. Choosing the tuned random forest is reasonable, not decisive.

## Run it

```bash
pip install -r requirements.txt
```

1. Download [IBM Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) and put it next to the notebooks.
2. Run the EDA notebook first. It writes `telco_churn_processed.csv`.
3. Run the modeling notebook. It saves `churn_model.pkl`.

### Using the saved model

`churn_model.pkl` is a dict with `model`, `scaler`, `features`, `best_model_name`, `optimal_threshold` and `summary_metrics`. It was saved with scikit-learn 1.8.0, so use the same version.

```python
import pickle, pandas as pd

with open("churn_model.pkl", "rb") as f:   # only unpickle files you trust
    bundle = pickle.load(f)

X = pd.DataFrame(...)[bundle["features"]]  # one-hot encoded, same columns as training
proba = bundle["model"].predict_proba(bundle["scaler"].transform(X))[:, 1]
churn_flag = proba >= 0.5
```

## Limitations and next steps

- **Threshold:** the saved `optimal_threshold` (0.606) was picked on the test set, using gradient boosting's probabilities, while the saved model is the tuned random forest. Treat 0.5 as the honest default until a threshold is chosen on validation data for the model that ships.
- **Cross-validation:** the tuning search's CV F1 (0.957) is inflated because resampled rows appear in both folds, so it is not quoted above. Section 20's 5-fold CV runs on the test set only. The fix is an `imblearn` pipeline with SMOTEENN inside each fold.
- SHAP explanations are wired in but need `pip install shap` to render.

## Credits

Dataset: IBM Telco Customer Churn (Kaggle, `blastchar`). The EDA structure started from a public tutorial and was extended; see the notebook headers.
