#!/bin/bash
set -e

# ================= CONFIGURATION =================
REMOTE_HOST=""
REMOTE_USER="dev"
REMOTE_DIR="/var/home/dev/workspace"
SSH_KEY="~/.ssh/id_ed25519"
# =================================================

echo "Syncing Script to ${REMOTE_HOST}..."

# Sync ONLY the folders mounted as volumes in docker-compose
rsync -avz -e "ssh -i $SSH_KEY" \
    --relative \
    --delete \
    dags/ \
    src/ \
	data/ \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

echo "✅ Sync Complete! Airflow should see changes immediately."
