#!/bin/bash
set -e
cd "$(dirname "$0")"

MODELS="llama3b llama8b qwen3b qwen7b qwen14b qwen32b"
START=$SECONDS

elapsed() {
  local secs=$1
  printf "%dh %dm %ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

echo "Started at $(date)"
echo ""

echo "Cleaning stale logs..."
rm -f logs/accuracy_bss/log.jsonl logs/dss/log.jsonl

echo "[$(date '+%H:%M:%S')] Starting accuracy_bss [GPUs 0-3] and dss [GPUs 4-7] in parallel..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python multiagent-debate.py \
  -e accuracy_bss -m $MODELS \
  --data_csv data_for_dss.csv --knowledge_flags_path knowledge_flags.json \
  --use_bss_scores --bss_scores_path accuracy_scores.json \
  > logs/accuracy_bss/stdout.log 2>&1 &
PID_ACC=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 python multiagent-debate.py \
  -e dss -m $MODELS \
  --data_csv data_for_dss.csv --knowledge_flags_path knowledge_flags.json \
  --use_dss_scores --bss_scores_path bss_scores_final.json \
  --alpha 0.2 --beta 0.0 \
  > logs/dss/stdout.log 2>&1 &
PID_DSS=$!

echo "  accuracy_bss PID=$PID_ACC  [GPUs 0-3]"
echo "  dss          PID=$PID_DSS  [GPUs 4-7]"

wait $PID_ACC
echo "[$(date '+%H:%M:%S')] accuracy_bss finished ($(elapsed $((SECONDS - START))))"

wait $PID_DSS
echo "[$(date '+%H:%M:%S')] dss finished ($(elapsed $((SECONDS - START))))"

echo ""
echo "[$(date '+%H:%M:%S')] Running evaluation on all experiments..."
python evaluate.py --all
echo "[$(date '+%H:%M:%S')] Evaluation done."

TOTAL=$((SECONDS - START))
echo ""
echo "All done at $(date) — total: $(elapsed $TOTAL)"
