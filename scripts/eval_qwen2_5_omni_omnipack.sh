#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ========== Task list ==========
tasks_list=(
    "dailyomni"
    "worldsense"
    "avutbenchmark"
    "videomme"
    "lvomnibench"
)

# ========== Overall token retention ratio ==========
ratio_pairs=(
    "0.10 0.275"   # R=25->R=12.5
    "0.075 0.25"   # R=20->R=10
    "0.05 0.225"   # R=15->R=7.5
    "0.03 0.175"   # R=10->R=5
)

tasks_str="${tasks_list[*]}"
ratios_str="$(IFS=';'; echo "${ratio_pairs[*]}")"

bash base/eval_qwen2_5_omni_zip.sh \
    --zip-method "omnipack" \
    --config "omnipack/config.yaml" \
    --tasks  "${tasks_str}" \
    --ratios "${ratios_str}" \
    "$@"