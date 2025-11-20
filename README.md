# Loan_default
MLOPs Project


1. Clone Repo

    git clone https://github.com/PondSterlingZx/Loan_default.git
    
    cd Loan_default



2. Install requirements


    pip install -r requirements.txt


3. Start Airflow


    docker-compose up -d


Then access Airflow at:

    http://localhost:8080
    username: admin
    password: admin

4. Run DAG once to generate processed data

    After triggering the DAG, the following files will be created:

    data/processed/clean_loan_default.csv
    
    data/processed/train.csv
    
    data/processed/test.csv
