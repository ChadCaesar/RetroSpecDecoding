# !/bin/bash

if [ $# -ne 6 ]; then
    echo "Usage: $0 <model_name> $1 <task_name> $2 <attn_type> $3 <dtype> $4 <budget_ratio> $5 <estimate_ratio>"
    exit 1
fi

NUM_EXAMPLES=${NUM_EXAMPLES:--1}
MODEL=${1}
TASK=${2}
ATTN_TYPE=${3}
DTYPE=${4}
BUDGET_RATIO=${5}
ESTIMATE_RATIO=${6}

MIN_DRAFT_STRIDE=${MIN_DRAFT_STRIDE:-3}
MAX_DRAFT_STRIDE=${MAX_DRAFT_STRIDE:-9}
DRAFT_MARGIN_THRESHOLD=${DRAFT_MARGIN_THRESHOLD:--1.0}
DRAFT_MARGIN_DROP_THRESHOLD=${DRAFT_MARGIN_DROP_THRESHOLD:--1.0}
MAX_SPARSE_STRIDE=${MAX_SPARSE_STRIDE:-32}
SPARSE_STABILITY_THRESHOLD=${SPARSE_STABILITY_THRESHOLD:--1.0}

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"
RESULT_DIR_E="./results/pred_e/${MODEL}/${ATTN_TYPE}"

echo "remove previous result file..."
rm -f "${RESULT_DIR}/${TASK}.jsonl"
rm -f "${RESULT_DIR_E}/${TASK}.jsonl"

echo "Start to predict..."
# numactl --cpunodebind=0,1 python -u pred.py \
python -u pred.py \
    --task ${TASK} \
    --attn_type ${ATTN_TYPE} \
    --model ${MODEL} \
    --dtype ${DTYPE} \
    --device auto \
    --retrieval_budget ${BUDGET_RATIO} \
    --estimation_budget ${ESTIMATE_RATIO} \
    --num_examples ${NUM_EXAMPLES} \
    --min_draft_stride ${MIN_DRAFT_STRIDE} \
    --max_draft_stride ${MAX_DRAFT_STRIDE} \
    --draft_margin_threshold ${DRAFT_MARGIN_THRESHOLD} \
    --draft_margin_drop_threshold ${DRAFT_MARGIN_DROP_THRESHOLD} \
    --max_sparse_stride ${MAX_SPARSE_STRIDE} \
    --sparse_stability_threshold ${SPARSE_STABILITY_THRESHOLD}
