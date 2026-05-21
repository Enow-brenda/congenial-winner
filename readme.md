# Female Autonomy in Family Planning — Classification & Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-green?style=flat-square)

> A machine learning pipeline and interactive dashboard for predicting female decision-making autonomy in family planning, based on the Cameroon Demographic and Health Survey (DHS) dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Getting Started](#getting-started)
- [Dashboard Features](#dashboard-features)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Project Overview

This project investigates the determinants of female autonomy in family planning decision-making using data from the **2018 Cameroon DHS (CMIR71FL)**. It combines rigorous statistical preprocessing, comparative model evaluation, and an interactive web application for real-time prediction.

The work is structured around two complementary components:

- **Classification pipeline** — trains and evaluates seven machine learning classifiers to predict a respondent's autonomy category based on socioeconomic, demographic, and healthcare access features.
- **Regression workspace** — exploratory analysis notebooks used for variable selection, recoding, and understanding relationships between predictors and outcomes.

---

## Live Demo

👉 [**Open the Streamlit App**](https://macgpt-congenial-winner.streamlit.app)

The app supports both **English** and **French** interfaces.

---

## Repository Structure

```
TP1-Regression/
│
├── data/
│   └── CMIR71FL.SAV                  # Raw DHS dataset (SPSS format)
│
├── women_autonomy_classifier/
│   ├── streamlit_app.py              # Interactive Streamlit dashboard
│   ├── training.py                   # Model training and evaluation pipeline
│   ├── config.py                     # Label maps and outcome definitions
│   ├── requirements.txt              # Python dependencies
│   ├── model_results.json            # Saved metrics and confusion matrix
│   ├── best_model.pkl                # Best performing model artifact
│   ├── preprocessor.pkl              # Fitted preprocessing pipeline
│   ├── label_encoder.pkl             # Fitted label encoder
│   └── old/
│       └── train_model.ipynb         # Legacy notebook training workflow
│
└── women_autonomy_regression/
    ├── final_regression.ipynb        # Final regression models and interpretation
    ├── variables_selection.ipynb     # Feature selection and variable recoding
    └── outputs/
        └── recoded_analysis_data.csv # Cleaned dataset used by training pipeline
```

---

## Methodology

### Data Source
The **2018 Cameroon DHS** (Standard DHS, Phase 7) provides individual-level data on women aged 15–49, covering reproductive health, contraceptive use, household wealth, education, and healthcare access.

### Target Variable
`fp_decision_autonomy` — a categorical variable encoding the level of female autonomy in family planning decisions, derived and recoded from the raw DHS indicators.

### Feature Engineering
Predictors include:

| Category | Features |
|---|---|
| Demographic | Age, number of children, marital status, marriage type |
| Socioeconomic | Education (woman & husband), wealth index, employment status |
| Geographic | Region, residence type (urban/rural) |
| Healthcare access | ANC visits, fieldworker contact, facility access, contraceptive method |
| Media exposure | Access to any media source |

### Model Comparison
Seven classifiers are evaluated using **5-fold stratified cross-validation**:

| Model | |
|---|---|
| Logistic Regression | AdaBoost |
| Decision Tree | Support Vector Machine (SVM) |
| Random Forest | LightGBM |
| XGBoost | |

The best-performing model is selected and saved as `best_model.pkl` for deployment.

---

## Getting Started

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/MacGPT237/congenial-winner.git
cd TP1-Regression
```

### 2. Set up environment

```bash
cd women_autonomy_classifier
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Prepare the dataset

Ensure the preprocessed dataset is available at:
```
women_autonomy_regression/outputs/recoded_analysis_data.csv
```

### 4. Train the model

```bash
python training.py
```

This generates the following artifacts in `women_autonomy_classifier/`:

| File | Description |
|---|---|
| `best_model.pkl` | Best classifier selected by cross-validation |
| `preprocessor.pkl` | Fitted imputation, scaling, and encoding pipeline |
| `label_encoder.pkl` | Fitted label encoder for target variable |
| `model_results.json` | Evaluation metrics and confusion matrix |

### 5. Launch the dashboard

```bash
streamlit run streamlit_app.py
```

---

## Dashboard Features

The Streamlit application provides four main tabs:

- **Prediction** — interactive form for entering respondent characteristics and obtaining a real-time autonomy category prediction with class probabilities.
- **Model Performance** — accuracy metrics, confusion matrix, and cross-validation results for all evaluated classifiers.
- **Feature Importance** — visualization of the most influential predictors driving model decisions.
- **About** — research context, dataset description, and project documentation.

---

## Dependencies

| Package | Version |
|---|---|
| `streamlit` | 1.57.0 |
| `scikit-learn` | 1.8.0 |
| `xgboost` | latest |
| `lightgbm` | latest |
| `imbalanced-learn` | latest |
| `pandas` | latest |
| `numpy` | latest |
| `matplotlib` | latest |
| `seaborn` | latest |
| `plotly` | latest |
| `joblib` | latest |

Full pinned versions available in `women_autonomy_classifier/requirements.txt`.

---

## Author

**Enow Brenda** — Master's in Data science
*Academic project — Statistics and Regression, 2025–2026*

---

> **Note:** This project is intended for academic demonstration purposes. The model artifacts are version-sensitive . if retraining, ensure the scikit-learn version matches the one specified in `requirements.txt`.