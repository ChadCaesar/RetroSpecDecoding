#!/bin/bash
set -o pipefail

export NUM_EXAMPLES=2
export MIN_DRAFT_STRIDE=1
export MAX_DRAFT_STRIDE=16
export DRAFT_MARGIN_THRESHOLD=0.25
export DRAFT_MARGIN_DROP_THRESHOLD=0.89
export MAX_SPARSE_STRIDE=64
export SPARSE_STABILITY_THRESHOLD=1.0

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

mkdir -p logs

TASK=passage_retrieval_en

for mode in Full_Flash_Attn RetroInfer SpecDecoder
do
    echo "Running task=${TASK}, mode=${mode}"

    bash pred.sh \
    llama-3-8b-1048k ${TASK} ${mode} bf16 0.018 0.232 \
    2>&1 | tee "logs/${TASK}_smoke_${mode}.log"

    status=${PIPESTATUS[0]}

    if [ ${status} -ne 0 ]; then
        echo "FAILED: task=${TASK}, mode=${mode}, status=${status}"
        exit ${status}
    fi
done

for mode in Full_Flash_Attn RetroInfer SpecDecoder
do
    python eval.py \
        --model llama-3-8b-1048k \
        --attn_type ${mode}
done

echo "LongBench smoke test completed."