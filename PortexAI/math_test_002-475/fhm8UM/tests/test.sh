#!/bin/bash

uv run /tests/portex_grade.py

if [ $? -ne 0 ]; then
  echo '{"total_score": 0.0, "error": "portex_grade.py failed"}' > /logs/verifier/reward.json
fi