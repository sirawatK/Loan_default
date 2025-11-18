FROM apache/airflow:2.7.1

# Copy source code first
USER root
COPY src /opt/airflow/src

# Switch to airflow user BEFORE pip install
USER airflow

# Install python dependencies safely under airflow user
RUN pip install --user --upgrade pip && \
    pip install --user pandas numpy scikit-learn

# Switch back to root to create the symlink in site-packages
USER root
RUN ln -s /opt/airflow/src /usr/local/lib/python3.8/site-packages/src

# Back to airflow user
USER airflow
