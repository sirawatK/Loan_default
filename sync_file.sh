#!/bin/bash
set -e

REMOTE_HOST=""
REMOTE_USER="dev"
REMOTE_DIR="/var/home/dev/workspace"
SSH_KEY="~/.ssh/id_ed25519"

echo "🚀 Syncing Code to ${REMOTE_HOST}..."

rsync -avz -e "ssh -i $SSH_KEY" \
    --relative \
    --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.env' \
    --exclude '.DS_Store' \
    dags/ \
    src/ \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

echo "✅ Sync Complete! Airflow & FastAPI should see new code immediately."
echo "⚠️  Note: 'data/' folder was NOT synced to prevent data loss."
