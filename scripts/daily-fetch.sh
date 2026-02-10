#!/usr/bin/env bash
set -eu

cd "/Users/ai/dev/peer-valuation-monitor"
PYTHON="/Users/ai/homebrew/opt/python@3.12/libexec/bin/python3"
LOG="[$(date +'%Y-%m-%d %H:%M:%S')]"

echo "$LOG Starting daily fetch..."

# Step 1: Fetch data + snapshot + changes (all in one)
$PYTHON fetch_data.py

# Step 2: Git commit + push
git add data/
git diff --cached --quiet && { echo "$LOG No changes to commit"; exit 0; }

git commit -m "Daily valuation snapshot: $(date +'%Y-%m-%d')"
git push origin main

echo "$LOG Daily fetch completed successfully"
