#!/bin/bash
#
# Run all 6 experiments: (a)-(f), two at a time on separate GPU sets.
#
# GPU allocation:
#   GPUs 0-3 (4x A6000, 192GB) — odd jobs
#   GPUs 4-7 (4x A40, 184GB)   — even jobs
#
# Usage:
#   bash run_all_experiments.sh

set -e

MODELS="llama3b llama8b qwen3b qwen7b qwen14b qwen32b"
BSS_DATA_CSV="data_for_bss.csv"
DSS_DATA_CSV="data_for_dss.csv"
BSS_SCORES="bss_scores_final.json"
RANDOM_SCORES="random_scores.json"
ACCURACY_SCORES="accuracy_scores.json"
KNOWLEDGE_FLAGS="knowledge_flags.json"

cd "$(dirname "$0")"

# --- Timing helpers ---
PIPELINE_START=$SECONDS

elapsed() {
  local secs=$1
  printf "%dh %dm %ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

step_start() {
  STEP_START=$SECONDS
  echo ""
  echo "────────────────────────────────────────────────"
  echo "[$(date '+%H:%M:%S')] $1"
  echo "────────────────────────────────────────────────"
}

step_done() {
  local dur=$((SECONDS - STEP_START))
  local total=$((SECONDS - PIPELINE_START))
  echo "[$(date '+%H:%M:%S')] $1 finished in $(elapsed $dur)  (total elapsed: $(elapsed $total))"
}

echo "=== End-to-end pipeline ==="
echo "Started at $(date)"
echo ""

mkdir -p logs

# --- Step 0a: Generate datasets (if not already present) ---
step_start "Step 0a: Generate datasets"
if [ ! -f "$BSS_DATA_CSV" ] || [ ! -f "$DSS_DATA_CSV" ]; then
  python generate_bss_dss_data.py \
    --bss_per_subject 50 --dss_per_subject 50 \
    --bss_out $BSS_DATA_CSV --dss_out $DSS_DATA_CSV
  step_done "Step 0a"
else
  echo "  $BSS_DATA_CSV and $DSS_DATA_CSV already exist, skipping."
fi

# --- Step 0b: Compute BSS scores (if not already present) ---
step_start "Step 0b: Compute BSS scores (sequential, one model at a time)"
if [ ! -f "$BSS_SCORES" ]; then
  python bss_calc.py --data_csv $BSS_DATA_CSV \
    > logs/bss_calc_stdout.log 2>&1
  mv bss_scores_only.json $BSS_SCORES
  step_done "Step 0b"
else
  echo "  $BSS_SCORES already exists, skipping."
fi

# --- Step 0c: Pre-compute knowledge flags ---
step_start "Step 0c: Compute knowledge flags"
if [ ! -f "$KNOWLEDGE_FLAGS" ]; then
  python compute_knowledge_flags.py \
    --data_csv $DSS_DATA_CSV \
    -m $MODELS \
    -o $KNOWLEDGE_FLAGS
  step_done "Step 0c"
else
  echo "  $KNOWLEDGE_FLAGS already exists, skipping."
fi

echo ""
echo "=== Starting discussion experiments ==="
echo ""

# Create log dirs upfront
mkdir -p logs/baseline logs/random_bss logs/accuracy_bss logs/bss logs/dss logs/binary_bss

# Helper: run an experiment only if its log.jsonl doesn't exist yet.
# Usage: run_experiment <gpu_list> <experiment_name> [extra_args...]
run_experiment() {
  local gpus="$1"; shift
  local exp="$1"; shift
  if [ -f "logs/$exp/log.jsonl" ]; then
    echo "  ($exp) already complete, skipping."
    return 1  # signal: nothing launched
  fi
  CUDA_VISIBLE_DEVICES=$gpus python multiagent-debate.py \
    -e "$exp" \
    -m $MODELS \
    --data_csv $DSS_DATA_CSV \
    --knowledge_flags_path $KNOWLEDGE_FLAGS \
    "$@" \
    > "logs/$exp/stdout.log" 2>&1 &
  echo "  ($exp) PID=$!  [GPUs $gpus]"
  return 0
}

# --- Batch 1: (a) baseline + (b) random scores ---
step_start "Batch 1: (a) baseline + (b) random_bss"
PIDS=()
run_experiment 0,1,2,3 baseline && PIDS+=($!)
run_experiment 4,5,6,7 random_bss \
  --use_bss_scores --bss_scores_path $RANDOM_SCORES && PIDS+=($!)
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
step_done "Batch 1"

# --- Batch 2: (c) accuracy scores + (d) actual BSS ---
step_start "Batch 2: (c) accuracy_bss + (d) bss"
PIDS=()
run_experiment 0,1,2,3 accuracy_bss \
  --use_bss_scores --bss_scores_path $ACCURACY_SCORES && PIDS+=($!)
run_experiment 4,5,6,7 bss \
  --use_bss_scores --bss_scores_path $BSS_SCORES && PIDS+=($!)
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
step_done "Batch 2"

# --- Batch 3: (e) DSS + (f) binary flags ---
step_start "Batch 3: (e) dss + (f) binary_bss"
PIDS=()
run_experiment 0,1,2,3 dss \
  --use_dss_scores --bss_scores_path $BSS_SCORES \
  --alpha 0.2 --beta 0.0 && PIDS+=($!)
run_experiment 4,5,6,7 binary_bss \
  --use_binary_syco_flags --bss_scores_path $BSS_SCORES && PIDS+=($!)
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
step_done "Batch 3"

# --- Step 4: Evaluate all experiments ---
step_start "Step 4: Evaluate all experiments"
python evaluate.py --all
step_done "Step 4"

# --- Final summary ---
TOTAL=$((SECONDS - PIPELINE_START))
echo ""
echo "════════════════════════════════════════════════"
echo "  Pipeline complete at $(date)"
echo "  Total wall time: $(elapsed $TOTAL)"
echo "════════════════════════════════════════════════"
echo ""
echo "Logs:        logs/<experiment>/log.jsonl"
echo "Evaluations: logs/<experiment>/eval/"
echo "Comparison:  logs/comparison.csv"
