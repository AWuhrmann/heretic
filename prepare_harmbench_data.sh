#!/bin/bash
# Downloads HarmBench's official test-split behaviors and extracts the
# non-contextual ("standard" FunctionalCategory) ones into a plain-text
# file, one behavior per line -- the format config.harmbench.toml's
# bad_evaluation_prompts expects. Contextual behaviors are excluded since
# our HarmBench classifier prompt (evaluator.py) only implements the
# non-contextual template (no [CONTEXT] field).
set -euo pipefail

OUT_FILE="${1:-harmbench_behaviors.txt}"

curl -sL --max-time 30 \
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv" \
    | python3 -c '
import csv
import sys

reader = csv.DictReader(sys.stdin)
for row in reader:
    if row["FunctionalCategory"] == "standard":
        print(row["Behavior"])
' > "$OUT_FILE"

echo "Wrote $(wc -l < "$OUT_FILE") behaviors to $OUT_FILE"
