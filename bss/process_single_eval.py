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

def load_samples_from_cleaned_json(cleaned_json_path: str):
    """Load MMLU samples from cleaned JSON file and convert to MMLUSample objects."""
    from utils import MMLUSample
    
    if not os.path.exists(cleaned_json_path):
        return None
    
    try:
        with open(cleaned_json_path, "r") as f:
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
        samples_list.sort(key=lambda x: x[0])
        return [s[1] for s in samples_list] if samples_list else None
    except Exception as e:
        print(f"Warning: Could not load samples from cleaned JSON: {e}")
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

    # 2. Sycophant with Knowledge (using knowledge_flags from JSONL or inference)
    print("\n[2] Sycophant with Knowledge (S_syco_k)...")
    if os.path.exists(jsonl_path):
        print(f"  Using JSONL: {jsonl_path}")
    else:
        print(f"  Warning: JSONL not found at {jsonl_path}, computing without knowledge check")
    
    # Load MMLU samples if model_name is provided (for inference-based knowledge check)
    mmlu_samples_for_syco = None
    if args.model_name and os.path.exists(jsonl_path):
        # Try to load MMLU samples (reuse the same logic as confident_sycophancy)
        csv_dir = os.path.dirname(csv_path)
        possible_cleaned_paths = [
            os.path.join(script_dir, "mmlu_debate_metadata_expt3.jsonl"),
        ]
        
        for cleaned_path in possible_cleaned_paths:
            if os.path.exists(cleaned_path):
                print(f"  Loading samples from cleaned JSON: {cleaned_path}")
                mmlu_samples_for_syco = load_samples_from_cleaned_json(cleaned_path)
                if mmlu_samples_for_syco:
                    print(f"  Loaded {len(mmlu_samples_for_syco)} samples from cleaned JSON")
                    break
        
        # If not found, try loading from HuggingFace
        if mmlu_samples_for_syco is None:
            try:
                from utils import load_mmlu_from_hf, MMLUSample
                # Extract sample questions from JSONL to match exact samples
                sample_questions_map = {}  # question -> sample_index
                if os.path.exists(jsonl_path):
                    try:
                        with open(jsonl_path, "r") as f:
                            for line in f:
                                line_data = json.loads(line)
                                for sample_name, sample_data in line_data.items():
                                    if sample_data.get("metric") == "sycophant_with_knowledge":
                                        sample_obj = sample_data.get("sample", {})
                                        question = sample_obj.get("question")
                                        sample_idx = sample_data.get("sample_index")
                                        if question and sample_idx is not None:
                                            sample_questions_map[question] = sample_idx
                    except Exception:
                        pass
                
                if not sample_questions_map:
                    print("  Warning: Could not extract sample questions from JSONL")
                    mmlu_samples_for_syco = None
                else:
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
                    max_idx = max(matched_samples.keys()) if matched_samples else -1
                    mmlu_samples_for_syco = [matched_samples[i] for i in range(max_idx + 1) if i in matched_samples]
                    print(f"  Matched {len(mmlu_samples_for_syco)} samples from JSONL")
            except Exception as e:
                print(f"  Warning: Could not load MMLU samples: {e}")
                mmlu_samples_for_syco = None
    
    for model in models:
        df_model = df[df["model"] == model]
        # Use the same model for knowledge check inference
        model_name_for_check = args.model_name if args.model_name else None
        result = compute_sycophant_with_knowledge_score(
            df_model=df_model,
            jsonl_path=jsonl_path if os.path.exists(jsonl_path) else None,
            model_name=model_name_for_check,  # Use provided model_name for inference if available
            mmlu_samples=mmlu_samples_for_syco  # Use loaded samples for inference if available
        )
        if "S_syco_k" in result and not np.isnan(result["S_syco_k"]):
            results.append({
                "model": model,
                "metric": "sycophant_with_knowledge",
                "score": result["S_syco_k"]
            })
            # Determine note based on whether knowledge check was performed
            if model_name_for_check and mmlu_samples_for_syco:
                note = " (with inference-based knowledge check)"
            elif os.path.exists(jsonl_path):
                note = " (using knowledge_flags)" 
            else:
                note = " (no knowledge check)"
            count_info = ""
            if "_count_supported" in result and "_count_eligible" in result:
                count_info = f" ({result['_count_supported']}/{result['_count_eligible']} eligible said 'correct'"
                if "_count_total_wrong" in result:
                    count_info += f", {result['_count_total_wrong']} total wrong samples)"
            print(f"  {model:15s}: {result['S_syco_k']:.4f}{note}{count_info}")

    # 3. Confident Sycophancy (requires model inference)
    print("\n[3] Confident Sycophancy...")
    if args.model_name and os.path.exists(jsonl_path):
        print(f"  Computing confident sycophancy with model: {args.model_name}")
        
        # Try to load MMLU samples
        mmlu_samples = None
        
        # First, try to load from cleaned JSON (if available)
        # Look for cleaned JSON in the same directory as the CSV
        csv_dir = os.path.dirname(csv_path)
        possible_cleaned_paths = [
            # os.path.join(csv_dir, "mmlu_debate_metadata_bss_cleaned.json"),
            os.path.join(script_dir, "mmlu_debate_metadata_expt3.jsonl"),
        ]
        
        for cleaned_path in possible_cleaned_paths:
            if os.path.exists(cleaned_path):
                print(f"  Loading samples from cleaned JSON: {cleaned_path}")
                mmlu_samples = load_samples_from_cleaned_json(cleaned_path)
                if mmlu_samples:
                    print(f"  Loaded {len(mmlu_samples)} samples from cleaned JSON")
                    break
        
        # If not found, try loading from HuggingFace
        if mmlu_samples is None:
            try:
                from utils import load_mmlu_from_hf
                print("  Loading MMLU samples from HuggingFace...")
                mmlu_samples = load_mmlu_from_hf(
                    subjects=["elementary_mathematics", "professional_law", 
                             "machine_learning", "business_ethics", "high_school_biology"],
                    split="test",
                    max_items=100
                )
                print(f"  Loaded {len(mmlu_samples)} samples from HuggingFace")
            except Exception as e:
                print(f"  Warning: Could not load MMLU samples: {e}")
                print("  Skipping confident sycophancy computation")
                mmlu_samples = None
        
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