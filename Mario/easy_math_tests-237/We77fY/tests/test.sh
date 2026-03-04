#!/bin/bash

uv run /tests/portex_grade.py

if [ $? -ne 0 ]; then
  # Harbor expects reward.json values to be numeric only.
  echo '{"reward": 0.0}' > /logs/verifier/reward.json
  echo '{"error": "portex_grade.py failed"}' > /logs/verifier/portex_detail.json
fi
