#!/bin/bash

echo "=== Agent Submission ==="
cat /app/answer.txt 2>/dev/null || echo "(no submission found at /app/answer.txt)"
echo ""
echo "=== Grading ==="

uv run /tests/portex_grade.py

if [ $? -ne 0 ]; then
  # Harbor expects reward.json values to be numeric only.
  # Write debug info to portex_detail.json instead of reward.json strings.
  echo '{"reward": 0.0}' > /logs/verifier/reward.json
  echo '{"error": "portex_grade.py failed"}' > /logs/verifier/portex_detail.json
fi
