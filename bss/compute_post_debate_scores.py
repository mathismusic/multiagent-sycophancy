"""
Script to compute post-debate sycophancy scores.

Usage:
    python compute_post_debate_scores.py --experiment expt3
    python compute_post_debate_scores.py --experiment expt3 --experiment expt4
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from process_eval_results import (
    compute_agreement_rate_score,
    compute_sycophant_with_knowledge_score,
    compute_confident_sycophancy_score
)

def compute_scores_for_experiment(experiment_name, jsonl_path=None, model_name=None, mmlu_samples=None):
    """Compute post-debate scores for a given experiment."""
    csv_path = f"eval_results/mmlu_debate_metadata_{experiment_name}__EVAL/summary_all.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return None
    
    if jsonl_path is None:
        jsonl_path = f"eval_results/mmlu_debate_metadata_{experiment_name}.jsonl"
        if not os.path.exists(jsonl_path):
            jsonl_path = None
    
    print(f"\n{'='*70}")
    print(f"Computing Post-Debate Scores: {experiment_name}")
    print(f"{'='*70}")
    
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
                "experiment": experiment_name,
                "model": model,
                "metric": "agreement_rate",
                "score": result["S_agree"]
            })
            print(f"  {model:15s}: {result['S_agree']:.4f}")
    
    # 2. Sycophant with Knowledge
    print("\n[2] Sycophant with Knowledge (S_syco_k)...")
    for model in models:
        df_model = df[df["model"] == model]
        result = compute_sycophant_with_knowledge_score(
            df_model=df_model,
            jsonl_path=jsonl_path,
            model_name=model_name,
            mmlu_samples=mmlu_samples
        )
        if "S_syco_k" in result and not np.isnan(result["S_syco_k"]):
            results.append({
                "experiment": experiment_name,
                "model": model,
                "metric": "sycophant_with_knowledge",
                "score": result["S_syco_k"]
            })
            note = " (with knowledge check)" if model_name and mmlu_samples else " (no knowledge check)"
            print(f"  {model:15s}: {result['S_syco_k']:.4f}{note}")
    
    # 3. Confident Sycophancy (if JSONL and model provided)
    if jsonl_path and model_name and mmlu_samples:
        print("\n[3] Confident Sycophancy...")
        try:
            confident_results = compute_confident_sycophancy_score(
                jsonl_path=jsonl_path,
                model_name=model_name,
                mmlu_samples=mmlu_samples
            )
            for model, score in confident_results.items():
                if not np.isnan(score):
                    results.append({
                        "experiment": experiment_name,
                        "model": model,
                        "metric": "confident_sycophancy",
                        "score": score
                    })
                    print(f"  {model:15s}: {score:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("\n[3] Confident Sycophancy: Skipped (requires --model-name and --mmlu-samples)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compute post-debate sycophancy scores")
    parser.add_argument("--experiment", action="append", required=True,
                        help="Experiment name(s) (e.g., expt3, expt4)")
    parser.add_argument("--jsonl-path", type=str, default=None,
                        help="Path to JSONL file (default: auto-detect from experiment)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name for knowledge check and confident_sycophancy")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for inference (default: auto)")
    parser.add_argument("--output", type=str, default="eval_results/post_debate_scores.csv",
                        help="Output CSV file path")
    
    args = parser.parse_args()
    
    # Load MMLU samples if model_name is provided
    mmlu_samples = None
    if args.model_name:
        try:
            from utils import load_mmlu_from_hf
            print("Loading MMLU samples for knowledge check...")
            mmlu_samples = load_mmlu_from_hf(
                subjects=["elementary_mathematics", "professional_law", 
                         "machine_learning", "business_ethics", "high_school_biology"],
                split="test",
                max_items=100
            )
            print(f"Loaded {len(mmlu_samples)} MMLU samples")
        except Exception as e:
            print(f"Warning: Could not load MMLU samples: {e}")
            print("  Continuing without knowledge check...")
    
    # Compute scores for each experiment
    all_results = []
    for exp in args.experiment:
        results = compute_scores_for_experiment(
            experiment_name=exp,
            jsonl_path=args.jsonl_path,
            model_name=args.model_name,
            mmlu_samples=mmlu_samples
        )
        if results:
            all_results.extend(results)
    
    # Save results
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(args.output, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to: {args.output}")
        print(f"{'='*70}")
        
        # Show summary
        print("\nSummary:")
        pivot = df_results.pivot_table(
            index="model", 
            columns=["experiment", "metric"], 
            values="score"
        )
        print(pivot.to_string())
    else:
        print("\nNo results computed.")

if __name__ == "__main__":
    main()

