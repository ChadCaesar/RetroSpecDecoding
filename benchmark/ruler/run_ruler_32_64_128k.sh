#!/bin/bash
set -o pipefail

export NUM_SAMPLES=5
export MIN_DRAFT_STRIDE=1
export MAX_DRAFT_STRIDE=16
export DRAFT_MARGIN_THRESHOLD=0.25
export DRAFT_MARGIN_DROP_THRESHOLD=0.89
export MAX_SPARSE_STRIDE=64
export SPARSE_STABILITY_THRESHOLD=1.0

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs

for task in niah_multiquery vt fwe qa_2
do
    for mode in Full_Flash_Attn RetroInfer SpecDecoder
    do
        echo "Running task=${task}, mode=${mode}, length=131072"

        bash ruler_run.sh \
        llama-3-8b-1048k full ${mode} 131072 ${task} bf16 0.018 0.232 \
        2>&1 | tee "logs/${task}_128k_${mode}.log"

        status=${PIPESTATUS[0]}

        if [ ${status} -ne 0 ]; then
            echo "FAILED: task=${task}, mode=${mode}, status=${status}"
            exit ${status}
        fi
    done
done

echo "All 64K experiments completed."