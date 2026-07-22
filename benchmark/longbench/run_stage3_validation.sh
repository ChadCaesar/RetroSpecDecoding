#!/usr/bin/env bash

set -Eeuo pipefail

trap 'echo "[ERROR] Failed at line ${LINENO}. Fix the error and rerun this script; completed outputs will be skipped."' ERR

# ============================================================
# Experiment configuration
# ============================================================

MODEL="llama-3-8b-1048k"
DTYPE="bf16"
DEVICE="auto"

START_INDEX=5
NUM_EXAMPLES=15
END_INDEX=$((START_INDEX + NUM_EXAMPLES - 1))

RETRIEVAL_BUDGET=0.018
ESTIMATION_BUDGET=0.232

MIN_DRAFT_STRIDE=1
MAX_DRAFT_STRIDE=16
DRAFT_MARGIN_THRESHOLD=0.25
DRAFT_MARGIN_DROP_THRESHOLD=0.89
MAX_SPARSE_STRIDE=64
SPARSE_STABILITY_THRESHOLD=2.0

TASKS=(
    passage_retrieval_en
    qasper
    2wikimqa
    gov_report
    repobench-p
)

MODES=(
    Full_Flash_Attn
    RetroInfer
    SpecDecoder
)

STABILITY_TAG="${SPARSE_STABILITY_THRESHOLD/./p}"

RUN_NAME="stage3_validation_idx${START_INDEX}_${END_INDEX}_stability_${STABILITY_TAG}_5tasks_${NUM_EXAMPLES}samples"
ARTIFACT_DIR="analysis_longbench/${RUN_NAME}"
LOG_DIR="${ARTIFACT_DIR}/logs"

FORCE_RERUN="${FORCE_RERUN:-0}"
OFFLINE="${OFFLINE:-0}"

mkdir -p "${LOG_DIR}"
mkdir -p "${ARTIFACT_DIR}/pred"
mkdir -p "${ARTIFACT_DIR}/scores"

# ============================================================
# Environment
# ============================================================

if [[ "${OFFLINE}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
else
    unset HF_HUB_OFFLINE || true
    unset TRANSFORMERS_OFFLINE || true
    unset HF_DATASETS_OFFLINE || true
fi

# ============================================================
# Source-code checks
# ============================================================

echo "[CHECK] Checking LongBench integration..."

grep -q "sample_id" pred.py || {
    echo "pred.py does not save sample_id."
    exit 1
}

grep -q "token_ids" pred.py || {
    echo "pred.py does not save token_ids."
    exit 1
}

grep -q "SpecDecoder" eval.py || {
    echo "eval.py does not accept SpecDecoder."
    exit 1
}

echo "[CHECK] Source-code checks passed."

# ============================================================
# Manifest
# ============================================================

{
    echo "Experiment: LongBench stage 1"
    echo "Purpose: compare task score and token-level equivalence"
    echo "Started: $(date --iso-8601=seconds)"
    echo "Git commit: $(git rev-parse HEAD)"
    echo
    echo "MODEL=${MODEL}"
    echo "DTYPE=${DTYPE}"
    echo "DEVICE=${DEVICE}"
    echo "START_INDEX=${START_INDEX}"
    echo "NUM_EXAMPLES=${NUM_EXAMPLES}"
    echo "END_INDEX=${END_INDEX}"
    echo "RETRIEVAL_BUDGET=${RETRIEVAL_BUDGET}"
    echo "ESTIMATION_BUDGET=${ESTIMATION_BUDGET}"
    echo "MIN_DRAFT_STRIDE=${MIN_DRAFT_STRIDE}"
    echo "MAX_DRAFT_STRIDE=${MAX_DRAFT_STRIDE}"
    echo "DRAFT_MARGIN_THRESHOLD=${DRAFT_MARGIN_THRESHOLD}"
    echo "DRAFT_MARGIN_DROP_THRESHOLD=${DRAFT_MARGIN_DROP_THRESHOLD}"
    echo "MAX_SPARSE_STRIDE=${MAX_SPARSE_STRIDE}"
    echo "SPARSE_STABILITY_THRESHOLD=${SPARSE_STABILITY_THRESHOLD}"
    echo "TASKS=${TASKS[*]}"
    echo "MODES=${MODES[*]}"
    echo
    echo "Git status:"
    git status --short
    echo
    echo "Python:"
    python --version
    echo
    echo "GPU:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free \
        --format=csv,noheader || true
} | tee "${ARTIFACT_DIR}/manifest.txt"

# ============================================================
# Output validation
# ============================================================

validate_output() {
    local output_file="$1"

    python - \
        "${output_file}" \
        "${START_INDEX}" \
        "${NUM_EXAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
start_index = int(sys.argv[2])
expected = int(sys.argv[3])
expected_sample_ids = list(
    range(start_index, start_index + expected)
)

if not path.exists():
    raise SystemExit(f"Missing output file: {path}")

rows = []
with path.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid JSON at {path}:{line_number}: {exc}"
            )

if len(rows) != expected:
    raise SystemExit(
        f"{path}: expected {expected} records, found {len(rows)}"
    )

sample_ids = [row.get("sample_id") for row in rows]
if None in sample_ids:
    raise SystemExit(f"{path}: missing sample_id")
try:
    sample_ids = [int(sample_id) for sample_id in sample_ids]
except (TypeError, ValueError):
    raise SystemExit(f"{path}: invalid sample_id values")
if sorted(sample_ids) != expected_sample_ids:
    raise SystemExit(f"{path}: expected sample_ids {expected_sample_ids[0]}-{expected_sample_ids[-1]}, but found {sorted(sample_ids)}")

for row in rows:
    if not isinstance(row.get("token_ids"), list):
        raise SystemExit(
            f"{path}: sample {row.get('sample_id')} has no token_ids list"
        )
    if not isinstance(row.get("pred"), str):
        raise SystemExit(
            f"{path}: sample {row.get('sample_id')} has invalid pred"
        )

print(f"[VALID] {path}: {expected} valid records")
PY
}

# ============================================================
# Prediction
# ============================================================

for task in "${TASKS[@]}"; do
    for mode in "${MODES[@]}"; do
        result_dir="results/pred/${MODEL}/${mode}"
        output_file="${result_dir}/${task}.jsonl"
        log_file="${LOG_DIR}/${task}_${mode}.log"

        mkdir -p "${result_dir}"

        line_count=0
        if [[ -f "${output_file}" ]]; then
            line_count=$(wc -l < "${output_file}")
        fi

        force_this_run="${FORCE_RERUN}"

        if [[ "${mode}" == "SpecDecoder" ]]; then
            force_this_run=1
        fi

        if [[ "${force_this_run}" != "1" && "${line_count}" -eq "${NUM_EXAMPLES}" ]]; then
            if validate_output "${output_file}"; then
                echo "[SKIP] ${task} / ${mode}: valid records for indices ${START_INDEX}-${END_INDEX} already exist."
                continue
            fi
            echo "[RERUN] Existing output does not match indices ${START_INDEX}-${END_INDEX}."
        fi

        if [[ -f "${output_file}" ]]; then
            echo "[REMOVE] Incomplete or obsolete output: ${output_file}"
            rm -f -- "${output_file}"
        fi

        echo
        echo "============================================================"
        echo "[RUN] Task=${task}, Mode=${mode}"
        echo "[PURPOSE] Measure official task score and equivalence to Full."
        echo "============================================================"

        python -u pred.py \
            --task "${task}" \
            --attn_type "${mode}" \
            --model "${MODEL}" \
            --dtype "${DTYPE}" \
            --device "${DEVICE}" \
            --retrieval_budget "${RETRIEVAL_BUDGET}" \
            --estimation_budget "${ESTIMATION_BUDGET}" \
            --start_index "${START_INDEX}" \
            --num_examples "${NUM_EXAMPLES}" \
            --min_draft_stride "${MIN_DRAFT_STRIDE}" \
            --max_draft_stride "${MAX_DRAFT_STRIDE}" \
            --draft_margin_threshold "${DRAFT_MARGIN_THRESHOLD}" \
            --draft_margin_drop_threshold "${DRAFT_MARGIN_DROP_THRESHOLD}" \
            --max_sparse_stride "${MAX_SPARSE_STRIDE}" \
            --sparse_stability_threshold "${SPARSE_STABILITY_THRESHOLD}" \
            2>&1 | tee "${log_file}"

        validate_output "${output_file}"
    done
done

# ============================================================
# Official LongBench evaluation
# ============================================================

for mode in "${MODES[@]}"; do
    echo
    echo "[EVAL] ${mode}"

    python -u eval.py \
        --model "${MODEL}" \
        --attn_type "${mode}" \
        2>&1 | tee "${LOG_DIR}/evaluate_${mode}.log"

    result_json="results/pred/${MODEL}/${mode}/result.json"

    if [[ ! -f "${result_json}" ]]; then
        echo "Missing evaluation result: ${result_json}"
        exit 1
    fi

    cp -- "${result_json}" \
        "${ARTIFACT_DIR}/scores/result_${mode}.json"
done

# ============================================================
# Copy relevant prediction files
# ============================================================

for mode in "${MODES[@]}"; do
    mkdir -p "${ARTIFACT_DIR}/pred/${mode}"

    for task in "${TASKS[@]}"; do
        cp -- \
            "results/pred/${MODEL}/${mode}/${task}.jsonl" \
            "${ARTIFACT_DIR}/pred/${mode}/${task}.jsonl"
    done
done

cp -- config/dataset2maxlen.json \
    "${ARTIFACT_DIR}/dataset2maxlen.json"

cp -- "$0" \
    "${ARTIFACT_DIR}/$(basename '$0')"

# ============================================================
# Token-level comparison with Full
# ============================================================

export ARTIFACT_DIR
export TASKS_CSV
export MODES_CSV

TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")
MODES_CSV=$(IFS=,; echo "${MODES[*]}")

export TASKS_CSV
export MODES_CSV

python - <<'PY'
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean

root = Path(os.environ["ARTIFACT_DIR"])
tasks = os.environ["TASKS_CSV"].split(",")
modes = os.environ["MODES_CSV"].split(",")

def load_jsonl(path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows

scores = {}
for mode in modes:
    path = root / "scores" / f"result_{mode}.json"
    with path.open("r", encoding="utf-8") as f:
        scores[mode] = json.load(f)

comparison_rows = []

for task in tasks:
    full_rows = load_jsonl(
        root / "pred" / "Full_Flash_Attn" / f"{task}.jsonl"
    )

    for mode in modes:
        mode_rows = load_jsonl(
            root / "pred" / mode / f"{task}.jsonl"
        )

        if set(full_rows) != set(mode_rows):
            raise SystemExit(
                f"sample_id mismatch: task={task}, mode={mode}"
            )

        for sample_id, full_row in full_rows.items():
            mode_row = mode_rows[sample_id]

            full_ids = full_row["token_ids"]
            mode_ids = mode_row["token_ids"]

            common = 0
            for full_token, mode_token in zip(full_ids, mode_ids):
                if full_token != mode_token:
                    break
                common += 1

            exact = full_ids == mode_ids
            first_diff = "" if exact else common
            prefix_ratio = (
                common / len(full_ids) if len(full_ids) > 0 else 1.0
            )

            comparison_rows.append({
                "task": task,
                "sample_id": sample_id,
                "mode": mode,
                "task_score": scores[mode].get(task),
                "exact_to_full": exact,
                "first_diff_position": first_diff,
                "common_prefix_tokens": common,
                "common_prefix_ratio": round(prefix_ratio, 6),
                "full_token_count": len(full_ids),
                "mode_token_count": len(mode_ids),
                "same_answers": (
                    full_row.get("answers") == mode_row.get("answers")
                ),
                "same_input_length": (
                    full_row.get("length") == mode_row.get("length")
                ),
            })

comparison_path = root / "sample_comparison.csv"
with comparison_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(comparison_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(comparison_rows)

grouped = defaultdict(list)
for row in comparison_rows:
    grouped[(row["task"], row["mode"])].append(row)

summary_rows = []
for task in tasks:
    for mode in modes:
        rows = grouped[(task, mode)]
        exact_count = sum(row["exact_to_full"] for row in rows)

        summary_rows.append({
            "task": task,
            "mode": mode,
            "samples": len(rows),
            "score": scores[mode].get(task),
            "exact_samples_vs_full": exact_count,
            "exact_rate_vs_full_percent": round(
                100 * exact_count / len(rows), 2
            ),
            "mean_common_prefix_ratio": round(
                fmean(row["common_prefix_ratio"] for row in rows), 6
            ),
            "same_answers": all(row["same_answers"] for row in rows),
            "same_input_length": all(
                row["same_input_length"] for row in rows
            ),
        })

summary_path = root / "task_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(summary_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"[SAVED] {comparison_path}")
print(f"[SAVED] {summary_path}")
PY

# ============================================================
# Package results
# ============================================================

{
    echo
    echo "Completed: $(date --iso-8601=seconds)"
} >> "${ARTIFACT_DIR}/manifest.txt"

BUNDLE="${ARTIFACT_DIR}.tar.gz"

tar -czf "${BUNDLE}" \
    -C "$(dirname "${ARTIFACT_DIR}")" \
    "$(basename "${ARTIFACT_DIR}")"

echo
echo "============================================================"
echo "[DONE] All experiments completed."
echo "Results: ${ARTIFACT_DIR}"
echo "Bundle:  ${BUNDLE}"
echo "============================================================"