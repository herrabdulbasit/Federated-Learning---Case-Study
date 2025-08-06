#!/bin/bash

# Set fixed values
CLIENTS=5
ROUNDS=3
SEED=42

# Define experiments as "strategy data"
EXPERIMENTS=(
  "sync iid"
  "sync non-iid-weak"
  "sync non-iid-medium"
  "sync non-iid-strong"
  "async iid"
  "async non-iid-weak"
  "async non-iid-medium"
  "async non-iid-strong"
  "hybrid iid"
)

for EXP in "${EXPERIMENTS[@]}"; do
  set -- $EXP
  STRATEGY=$1
  DATA=$2

  echo "▶Running experiment: strategy=$STRATEGY, data=$DATA"
  python main.py \
    --strategy $STRATEGY \
    --data $DATA

  echo "Completed: strategy=$STRATEGY, data=$DATA"
  echo "--------------------------------------------"
done

echo "🎉 All experiments finished."
