#!/bin/bash
#
# Run 8 experiments (all except warning_only) with a configurable seed.
# Results saved to logs_seed${SEED}/
#
# GPU allocation:
#   GPUs 0-3 (4x A6000, 192GB) — odd jobs
#   GPUs 4-7 (4x A40, 184GB)   — even jobs
#
# Usage:
#   bash run_all_experiments_seeded.sh <seed>
#   bash run_all_experiments_seeded.sh 123

set -e

SEED="${1:?Usage: $0 <seed>}"
LOGDIR="logs_seed${SEED}"

MODELS="llama3b llama8b qwen3b qwen7b qwen14b qwen32b"
DSS_DATA_CSV="data_for_dss.csv"
BSS_SCORES="bss_scores_final.json"
RANDOM_SCORES="random_scores.json"
ACCURACY_SCORES="accuracy_scores.json"
KNOWLEDGE_FLAGS="knowledge_flags.json"
DBSS_SCORES="bss_scores_post_debate.json"

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

echo "=== Seeded pipeline (seed=$SEED) ==="
echo "Started at $(date)"
echo "Logs: $LOGDIR/"
echo ""

# Create log dirs upfront
mkdir -p $LOGDIR/{baseline,bss,accuracy_bss,random_bss,dss,binary_bss,random_binary,dbss}

# Helper: run an experiment only if its log.jsonl doesn't exist yet.
# Usage: run_experiment <gpu_list> <experiment_name> [extra_args...]
run_experiment() {
  local gpus="$1"; shift
  local exp="$1"; shift
  if [ -f "$LOGDIR/$exp/log.jsonl" ]; then
    echo "  ($exp) already complete, skipping."
    return 1  # signal: nothing launched
  fi
  CUDA_VISIBLE_DEVICES=$gpus python multiagent-debate.py \
    -e "$exp" \
    -m $MODELS \
    --data_csv $DSS_DATA_CSV \
    --knowledge_flags_path $KNOWLEDGE_FLAGS \
    --seed $SEED \
    --log_dir $LOGDIR \
    "$@" \
    > "$LOGDIR/$exp/stdout.log" 2>&1 &
  echo "  ($exp) PID=$!  [GPUs $gpus]  seed=$SEED"
  return 0
}

# --- Batch 1: (a) baseline + (b) bss ---
step_start "Batch 1: (a) baseline + (b) bss"
PIDS=()
run_experiment 0,1,2,3 baseline && PIDS+=($!)
run_experiment 4,5,6,7 bss \
  --use_bss_scores --bss_scores_path $BSS_SCORES && PIDS+=($!)
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
step_done "Batch 1"

# --- Batch 2: (c) accuracy scores + (d) random scores ---
step_start "Batch 2: (c) accuracy_bss + (d) random_bss"
PIDS=()
run_experiment 0,1,2,3 accuracy_bss \
  --use_bss_scores --bss_scores_path $ACCURACY_SCORES && PIDS+=($!)
run_experiment 4,5,6,7 random_bss \
  --use_bss_scores --bss_scores_path $RANDOM_SCORES && PIDS+=($!)
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

# --- Batch 4: (g) random binary + (h) dbss ---
step_start "Batch 4: (g) random_binary + (h) dbss"
PIDS=()
run_experiment 0,1,2,3 random_binary \
  --use_random_binary_flags && PIDS+=($!)
run_experiment 4,5,6,7 dbss \
  --use_bss_scores --bss_scores_path $DBSS_SCORES && PIDS+=($!)
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
step_done "Batch 4"

# --- Evaluate all experiments ---
step_start "Evaluate all experiments"
for exp in baseline bss accuracy_bss random_bss dss binary_bss random_binary dbss; do
  python evaluate.py -e "$exp" --log_dir $LOGDIR
done
step_done "Evaluate"

# --- Final summary ---
TOTAL=$((SECONDS - PIPELINE_START))
echo ""
echo "════════════════════════════════════════════════"
echo "  Pipeline complete at $(date)"
echo "  Seed: $SEED"
echo "  Total wall time: $(elapsed $TOTAL)"
echo "════════════════════════════════════════════════"
echo ""
echo "Logs:        $LOGDIR/<experiment>/log.jsonl"
echo "Evaluations: $LOGDIR/<experiment>/eval/"
