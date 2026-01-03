from process_eval_results import (
    overall_from_csv,
    compute_agreement_rate_score,
    compute_sycophant_with_knowledge_score,
    compute_confident_sycophancy_score
)
import os
import pandas as pd
import numpy as np
import argparse
import json

def load_mmlu_samples_for_metric(jsonl_path: str, metric_name: str, script_dir: str):
    """
    Load MMLU samples for a given metric, matching questions from JSONL.
    Tries cleaned JSON first, then HuggingFace.
    """
    from utils import load_mmlu_from_hf, MMLUSample
    
    # First, try to load from cleaned JSON
    possible_cleaned_paths = [
        os.path.join(script_dir, "mmlu_debate_metadata_expt3.jsonl"),
    ]
    
    for cleaned_path in possible_cleaned_paths:
        if os.path.exists(cleaned_path):
            try:
                with open(cleaned_path, "r") as f:
                    data = json.load(f)
                
                # Extract samples and sort by sample_index
                samples_list = []
                for sample_name, sample_data in data.items():
                    if "sample" in sample_data and "sample_index" in sample_data:
                        sample_obj = sample_data["sample"]
                        sample_index = sample_data["sample_index"]
                        
                        # Convert to MMLUSample format
                        if "choices" in sample_obj and "correct_idx" in sample_obj:
                            mmlu_sample = MMLUSample(
                                question=sample_obj.get("question", ""),
                                choices=sample_obj["choices"],
                                correct_idx=sample_obj["correct_idx"],
                                subject=sample_obj.get("subject", "")
                            )
                            samples_list.append((sample_index, mmlu_sample))
                
                # Sort by sample_index and return just the samples
                if samples_list:
                    samples_list.sort(key=lambda x: x[0])
                    return [s[1] for s in samples_list]
            except Exception as e:
                print(f"  Warning: Could not load from cleaned JSON: {e}")
                continue
    
    # If not found, try loading from HuggingFace and match by question
    if not os.path.exists(jsonl_path):
        return None
    
    try:
        # Extract sample questions from JSONL to match exact samples
        sample_questions_map = {}  # question -> sample_index
        with open(jsonl_path, "r") as f:
            for line in f:
                line_data = json.loads(line)
                for sample_name, sample_data in line_data.items():
                    if sample_data.get("metric") == metric_name:
                        sample_obj = sample_data.get("sample", {})
                        question = sample_obj.get("question")
                        sample_idx = sample_data.get("sample_index")
                        if question and sample_idx is not None:
                            sample_questions_map[question] = sample_idx
        
        if not sample_questions_map:
            return None
        
        # Load MMLU samples and match by question text
        print(f"  Loading MMLU samples from HuggingFace (matching {len(sample_questions_map)} questions from JSONL)...")
        all_mmlu_samples = load_mmlu_from_hf(
            subjects=["elementary_mathematics", "professional_law", 
                     "machine_learning", "business_ethics", "high_school_biology"],
            split="test",
            max_items=None  # Load all to find matches
        )
        
        # Match samples by question and create ordered list by sample_index
        matched_samples = {}  # sample_index -> MMLUSample
        for mmlu_sample in all_mmlu_samples:
            if mmlu_sample.question in sample_questions_map:
                sample_idx = sample_questions_map[mmlu_sample.question]
                matched_samples[sample_idx] = mmlu_sample
        
        # Create ordered list (sorted by sample_index)
        if matched_samples:
            max_idx = max(matched_samples.keys())
            mmlu_samples = [matched_samples[i] for i in range(max_idx + 1) if i in matched_samples]
            print(f"  Matched {len(mmlu_samples)} samples from JSONL")
            return mmlu_samples
    except Exception as e:
        print(f"  Warning: Could not load MMLU samples: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Compute single eval results including confident sycophancy")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to summary_all.csv (default: auto-detect)")
    parser.add_argument("--jsonl-path", type=str, default=None,
                        help="Path to JSONL file (default: auto-detect)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name for confident_sycophancy computation (required for confident sycophancy)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for inference (default: auto)")
    parser.add_argument("--title", type=str, default="MMLU Debate Results",
                        help="Title for output table")
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default paths if not provided
    if args.csv_path is None:
        csv_path = os.path.join(script_dir, "mmlu_debate_metadata_use_bss_data_for_debate --use_bss_scores__EVAL/summary_all.csv")
    else:
        csv_path = args.csv_path if os.path.isabs(args.csv_path) else os.path.join(script_dir, args.csv_path)
    
    if args.jsonl_path is None:
        jsonl_path = os.path.join(script_dir, "eval_results/mmlu_debate_metadata (2)_exp4.jsonl")
    else:
        jsonl_path = args.jsonl_path if os.path.isabs(args.jsonl_path) else os.path.join(script_dir, args.jsonl_path)
    
    title = args.title

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

    # 2. Sycophant with Knowledge (check knowledge for each model separately)
    print("\n[2] Sycophant with Knowledge (S_syco_k)...")
    if os.path.exists(jsonl_path):
        print(f"  Using JSONL: {jsonl_path}")
        # Load MMLU samples
        mmlu_samples_for_syco = load_mmlu_samples_for_metric(jsonl_path, "sycophant_with_knowledge", script_dir)
        
        if mmlu_samples_for_syco:
            try:
                syco_results = compute_sycophant_with_knowledge_score(
                    jsonl_path=jsonl_path,
                    df=df,
                    mmlu_samples=mmlu_samples_for_syco,
                    device=args.device
                )
                for model, score in syco_results.items():
                    if not np.isnan(score):
                        results.append({
                            "model": model,
                            "metric": "sycophant_with_knowledge",
                            "score": score
                        })
                        print(f"  {model:15s}: {score:.4f} (with inference-based knowledge check)")
            except Exception as e:
                print(f"  Error computing sycophant_with_knowledge: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  Skipped: Could not load MMLU samples")
    else:
        print(f"  Warning: JSONL not found at {jsonl_path}, skipping knowledge check")

    # 3. Confident Sycophancy (requires model inference)
    print("\n[3] Confident Sycophancy...")
    if args.model_name and os.path.exists(jsonl_path):
        print(f"  Computing confident sycophancy with model: {args.model_name}")
        
        # Load MMLU samples (same function as sycophant_with_knowledge)
        mmlu_samples = load_mmlu_samples_for_metric(jsonl_path, "confident_sycophancy", script_dir)
        
        # Compute confident sycophancy if we have samples
        if mmlu_samples:
            try:
                confident_results = compute_confident_sycophancy_score(
                    jsonl_path=jsonl_path,
                    model_name=args.model_name,
                    mmlu_samples=mmlu_samples,
                    device=args.device
                )
                for model, score in confident_results.items():
                    if not np.isnan(score):
                        results.append({
                            "model": model,
                            "metric": "confident_sycophancy",
                            "score": score
                        })
                        print(f"  {model:15s}: {score:.4f}")
            except Exception as e:
                print(f"  Error computing confident sycophancy: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  Skipped: Could not load MMLU samples")
    else:
        if not args.model_name:
            print("  Skipped: --model-name not provided")
        if not os.path.exists(jsonl_path):
            print(f"  Skipped: JSONL not found at {jsonl_path}")
        print("  Note: Requires --model-name and JSONL file with debate_state data")

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

if __name__ == "__main__":
    main()