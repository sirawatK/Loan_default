# Loan_default
MLOPs Project


1 Clone Repo
git clone https://github.com/PondSterlingZx/Loan_default.git
cd Loan_default

2 Install requirements
pip install -r requirements.txt

3 Start Airflow
docker-compose up -d

then access 
http://localhost:8080
  login with
    username: admin
    password: admin


Run DAG once to get
  data/processed/clean_loan_default.csv
  data/processed/train.csv
  data/processed/test.csv
