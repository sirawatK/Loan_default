## Project Overview

Loan_default aims to build a full pipeline for predicting whether a loan will default or not. The project includes:

- Data ingestion & processing  
- Feature engineering & cleaning  
- Training machine-learning models  
- Evaluation and analysis  


This setup helps ensure clean data flow and reproducible model training — suitable for experimentation or production-like workflows.  

## Features

- Data processing pipeline (raw → processed)  
- Configurable machine-learning model training  
- Separation between data ingestion, processing, modeling for clarity & modularity  
- Automation via DAGs (for scheduling / reproducibility)  

## Repository Structure

```
├── data/
│   ├── raw/              # raw/unprocessed data files  
│   └── processed/        # cleaned and preprocessed data ready for modeling  
├── notebooks/            # Jupyter notebooks for EDA, experiments  
├── src/                  # source code: data processing, feature engineering, model definitions, utilities  
├── dags/                 # DAG definitions for data pipeline orchestration  
├── docker/               # Docker-related configs (if any)  
├── .env.example          # example environment variables (API keys, config, etc.)  
├── docker-compose.yaml   # for spinning up services (e.g. workflow engine)  
└── README.md             # this file  

## Getting Started

To get a copy of the project up and running on your local machine for development or testing:

```bash
git clone https://github.com/sirawatK/Loan_default.git
cd Loan_default
```

### Environment Setup

1. Create and activate a virtual environment (recommended).  
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set any needed environment variables (database credentials, API keys, etc.).  

## How to Run

To build the data and run the pipeline + modeling:

```bash
# start the workflow engine
docker-compose up -d

# open the UI of the workflow engine (e.g. Airflow) at:
# http://localhost:8080
# login with default credentials (or as configured)

# Trigger the DAG to process raw data:
# This will generate processed datasets, e.g.:
# data/processed/clean_loan_default.csv
# data/processed/train.csv
# data/processed/test.csv


## Data Pipeline (DAG)

The pipelines (defined in `dags/`) orchestrate the data flow:

- **Raw ingestion** — fetch / copy raw data into `data/raw/`  
- **Processing & cleaning** — transform raw data into a clean, feature-ready format stored in `data/processed/`  
- **Train/Test split** — split processed data for modeling and evaluation  

This structure ensures reproducibility and ease of updates when new data arrives.  

## Model Training & Prediction

You can train new models or re-train existing ones on the processed dataset (from `data/processed/`). After training, the model artifacts can be saved (e.g. as `.pkl` / serialized objects), and used for inference/prediction on new loan data.  

Notebooks under `notebooks/` provide EDA, feature exploration, modeling experiments, and evaluation metrics.  

