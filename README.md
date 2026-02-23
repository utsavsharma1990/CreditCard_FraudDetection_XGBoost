# Credit Card Fraud Detection using XGBoost

A machine learning project that detects fraudulent credit card transactions using **XGBoost** and other classifiers. Built on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains 284,807 transactions with only 492 (0.17%) being fraudulent — a highly imbalanced classification problem.

## Highlights

- **Multiple model comparison**: Logistic Regression, Random Forest, and XGBoost
- **Class imbalance handling**: SMOTE oversampling vs. XGBoost's `scale_pos_weight`
- **Hyperparameter tuning**: RandomizedSearchCV with stratified cross-validation
- **Threshold optimization**: finding the optimal classification threshold for F1-score
- **Model interpretability**: SHAP values and feature importance analysis
- **Comprehensive evaluation**: ROC-AUC, Precision-Recall curves, confusion matrices, and cross-validation

## Dataset

The dataset contains credit card transactions made by European cardholders in September 2013.

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Features | 30 (V1-V28 from PCA, Time, Amount) |
| Target | Class (0 = Normal, 1 = Fraud) |

> **Note:** The dataset is not included in this repository due to its size. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project root.

## Project Structure

```
CreditCard_FraudDetection_XGBoost/
├── Credit_Card_Fraud_Detection.ipynb    # Main notebook (comprehensive analysis)
├── Credit Card Fraud Detection.ipynb    # Original baseline notebook
├── requirements.txt                     # Python dependencies
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/utsavsharma1990/CreditCard_FraudDetection_XGBoost.git
cd CreditCard_FraudDetection_XGBoost
pip install -r requirements.txt
```

### Running

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it in the project root directory
3. Launch the notebook:

```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

## Approach

### 1. Exploratory Data Analysis
- Class distribution analysis (0.17% fraud rate)
- Transaction amount distributions by class
- Temporal pattern analysis
- Feature correlation with fraud target
- Top feature distribution comparisons (Normal vs. Fraud)

### 2. Preprocessing
- StandardScaler normalization on the Amount feature
- Time feature dropped after analysis
- V1-V28 features already PCA-transformed (no scaling needed)

### 3. Class Imbalance Handling
Two strategies compared:
- **SMOTE**: Synthetic Minority Oversampling Technique — generates synthetic fraud samples
- **`scale_pos_weight`**: XGBoost's built-in class weight adjustment (no synthetic data)

### 4. Models Trained
| Model | Imbalance Strategy |
|-------|--------------------|
| Logistic Regression | SMOTE |
| Random Forest | SMOTE |
| XGBoost (default) | SMOTE |
| XGBoost (weighted) | `scale_pos_weight` |
| XGBoost (tuned) | SMOTE + hyperparameter tuning |

### 5. Evaluation Metrics
For imbalanced fraud detection, **accuracy is misleading** (a naive "predict all normal" classifier achieves 99.83%). Instead, we focus on:

| Metric | Why It Matters |
|--------|----------------|
| **AUPRC** | Best single metric for imbalanced classification |
| **Recall** | Missed fraud has high financial cost |
| **Precision** | False alerts erode customer trust |
| **F1-Score** | Balanced precision-recall trade-off |
| **ROC-AUC** | Overall discriminative ability |

### 6. Threshold Optimization
Default 0.5 threshold is suboptimal for imbalanced data. The notebook searches for the threshold that maximizes F1-score and visualizes the precision-recall-threshold trade-off.

### 7. Model Interpretability
- **XGBoost Feature Importance** (gain-based)
- **SHAP Values** — shows per-feature directional impact on fraud predictions

## Key Takeaways

- XGBoost with hyperparameter tuning achieves the best balance of precision and recall
- SMOTE and `scale_pos_weight` are both effective strategies for handling extreme class imbalance
- Threshold tuning is critical for real-world deployment — business costs determine the optimal operating point
- SHAP analysis reveals which PCA components (and transaction amount) drive fraud predictions

## Technologies

- Python 3
- XGBoost
- scikit-learn
- imbalanced-learn (SMOTE)
- SHAP
- pandas / NumPy
- matplotlib / seaborn

## License

This project is open source and available under the [MIT License](LICENSE).
