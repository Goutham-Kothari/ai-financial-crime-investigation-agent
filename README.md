# AI Financial Crime Investigation Agent

An AI/ML portfolio project for financial transaction risk scoring and investigation using the PaySim synthetic mobile-money dataset.

## Overview

This project combines machine learning, rule-based evidence checks, a REST API, and an interactive dashboard to support the investigation of potentially suspicious transactions.

The system can:
- Score transactions using a trained Random Forest model.
- Examine transaction and account balance patterns.
- Present evidence and findings for analyst review.
- Provide a REST API through FastAPI.
- Display investigation results through a Streamlit dashboard.
- Compare a machine-learning model with a simple rule-based baseline.

**Important:** This is an educational prototype built with synthetic PaySim data. It is not a production banking or compliance system, and its results should not be interpreted as real-world fraud detection performance.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- FastAPI and Uvicorn
- Streamlit
- PaySim synthetic transaction dataset

## Project Structure

```text
financial-crime-investigation-agent/
├── data/
│   └── paysim.csv
├── models/
│   ├── paysim_random_forest.joblib
│   └── paysim_random_forest_v2.joblib
├── reports/
├── src/
│   ├── api.py
│   ├── train_model.py
│   ├── train_model_v2.py
│   ├── dataset_investigator.py
│   ├── investigate_transaction.py
│   ├── evaluate_model.py
│   ├── feature_analysis.py
│   └── ...
├── dashboard.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the PaySim CSV file at `data/paysim.csv` before running the data analysis or training scripts.

## Run the Application

Start the API in one terminal:

```powershell
uvicorn src.api:app --reload
```

Open the API documentation at:

`http://127.0.0.1:8000/docs`

Start the dashboard in a second terminal:

```powershell
python -m streamlit run dashboard.py
```

The dashboard should open at `http://localhost:8501`.

## Model Evaluation

The V2 Random Forest achieved perfect precision, recall, average precision, and ROC-AUC in the current chronological holdout experiment on PaySim. The test set contained 123,580 transactions, including 1,654 fraud cases.

A simple rule-based baseline achieved:
- Precision: 22.43%
- Recall: 99.52%
- False positives: 5,693
- False negatives: 8

These results are specific to the current PaySim experiment. The exceptionally strong V2 result requires further independent validation and may reflect patterns in the synthetic data. It does not demonstrate equivalent performance on real financial transactions.

## Responsible Use and Limitations

- PaySim is synthetic data, not real customer transaction data.
- Model scores indicate predicted risk, not proof of criminal activity.
- Rule-based findings are investigation signals, not definitive conclusions.
- Production deployment would require independent validation, monitoring, security controls, privacy safeguards, and human review.
- The model has not been validated for real banking or regulatory use.

## Future Improvements

- Evaluate on additional datasets and more rigorous holdout designs.
- Measure alert volume, precision-recall trade-offs, and calibration.
- Add analyst feedback and investigation audit trails.
- Improve explainability and test API input validation.
- Add automated tests and deployment documentation.

## Project Purpose

Built as a portfolio project to demonstrate practical skills in Python, machine learning, data analysis, REST APIs, and investigation-oriented application development.
=======
# ai-financial-crime-investigation-agent
An AI-powered financial crime investigation agent using machine learning to detect and investigate suspicious transactions.
