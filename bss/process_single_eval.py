from process_eval_results import (
    overall_from_csv,
    compute_agreement_rate_score,
    compute_sycophant_with_knowledge_score,
    compute_confident_sycophancy_score
)
import os
import pandas as pd
import numpy as np

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "mmlu_debate_metadata_use_bss_data_for_debate --use_bss_scores__EVAL/summary_all.csv")
jsonl_path = os.path.join(script_dir, "eval_results/mmlu_debate_metadata (2)_exp4.jsonl")

title = "MMLU Debate Results (Iterated)"

# Compute old metrics (overall sycophancy)
print("="*70)
print("Computing Overall Sycophancy (Old Metrics)")
print("="*70)
df_old = overall_from_csv(csv_path, title)
print(df_old.to_string(index=False))

# Compute new metrics (post-debate scores)
print("\n" + "="*70)
print("Computing Post-Debate Scores (New Metrics)")
print("="*70)

# Load CSV
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

models = sorted(df["model"].unique())
results = []

# 1. Agreement Rate
print("\n[1] Agreement Rate (S_agree)...")
for model in models:
    df_model = df[df["model"] == model]
    result = compute_agreement_rate_score(df_model)
    if "S_agree" in result and not np.isnan(result["S_agree"]):
        results.append({
            "model": model,
            "metric": "agreement_rate",
            "score": result["S_agree"]
        })
        count_info = ""
        if "_count_supported" in result and "_count_total" in result:
            count_info = f" ({result['_count_supported']}/{result['_count_total']} said 'correct')"
        print(f"  {model:15s}: {result['S_agree']:.4f}{count_info}")

# 2. Sycophant with Knowledge (using knowledge_flags from JSONL)
print("\n[2] Sycophant with Knowledge (S_syco_k)...")
if os.path.exists(jsonl_path):
    print(f"  Using knowledge_flags from JSONL: {jsonl_path}")
else:
    print(f"  Warning: JSONL not found at {jsonl_path}, computing without knowledge check")
for model in models:
    df_model = df[df["model"] == model]
    result = compute_sycophant_with_knowledge_score(
        df_model=df_model,
        jsonl_path=jsonl_path if os.path.exists(jsonl_path) else None,
        model_name=None,  # Not needed if using knowledge_flags
        mmlu_samples=None  # Not needed if using knowledge_flags
    )
    if "S_syco_k" in result and not np.isnan(result["S_syco_k"]):
        results.append({
            "model": model,
            "metric": "sycophant_with_knowledge",
            "score": result["S_syco_k"]
        })
        note = " (using knowledge_flags)" if os.path.exists(jsonl_path) else " (no knowledge check)"
        count_info = ""
        if "_count_supported" in result and "_count_eligible" in result:
            count_info = f" ({result['_count_supported']}/{result['_count_eligible']} eligible said 'correct'"
            if "_count_total_wrong" in result:
                count_info += f", {result['_count_total_wrong']} total wrong samples)"
        print(f"  {model:15s}: {result['S_syco_k']:.4f}{note}{count_info}")

# 3. Confident Sycophancy (requires model inference - can't use pre-computed flags)
print("\n[3] Confident Sycophancy...")
print("  Note: Requires model inference (needs model_name and mmlu_samples)")
print("  Unlike sycophant_with_knowledge, this can't use pre-computed flags because")
print("  it needs to compute P('correct') vs P('incorrect') on round 5 prompts.")
print("  Use compute_post_debate_scores.py with --model-name for full computation")

# Display results table
if results:
    print("\n" + "="*70)
    print("New Metrics Summary")
    print("="*70)
    df_new = pd.DataFrame(results)
    pivot = df_new.pivot_table(
        index="model",
        columns="metric",
        values="score"
    )
    print(pivot.to_string())
else:
    print("\nNo new metrics computed.")