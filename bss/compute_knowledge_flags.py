"""
Pre-compute knowledge probes for each (sample, model) pair.

For every sample in the dataset and every model, queries the model on a
neutral MCQ prompt (no user stance / no discussion context) and records
the model's argmax option letter (A/B/C/D).

Output: a JSON file keyed by sample index (as string), each value is a dict
mapping model name to the argmax letter (e.g. "A", "B", "C", "D").

Consumers resolve this to a bool at runtime by comparing against the
specific wrong stance chosen for that sample:
    knows = (knowledge_flags[model] != wrong_letter)

Arguments:
  --data_csv      (required) Path to the dataset CSV (e.g. data_for_dss.csv)
  --models, -m    (required) Space-separated list of model short names
                  (e.g. llama3b llama8b qwen3b qwen7b qwen14b qwen32b)
  --output, -o    Output JSON path (default: knowledge_flags.json)
  --device        Torch device (default: auto)

Usage:
  python compute_knowledge_flags.py --data_csv data_for_dss.csv \\
      -m llama3b llama8b qwen3b qwen7b qwen14b qwen32b \\
      -o knowledge_flags.json
"""

import argparse
import json
import os

from utils import load_mmlu_from_csv, models_map, _format_mcq_prompt, _correct_letter_for_sample
from prompt import MMLU_SYSTEM_PROMPT_KNOWLEDGE
from logprobs_model import LogprobsModel


def main():
    parser = argparse.ArgumentParser(description="Pre-compute knowledge probes for (sample, model) pairs.")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--models", "-m", nargs="+", required=True, help="Model short names")
    parser.add_argument("--output", "-o", type=str, default="knowledge_flags.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="auto", help="Torch device")
    args = parser.parse_args()

    samples = load_mmlu_from_csv(args.data_csv)
    print(f"Loaded {len(samples)} samples from {args.data_csv}")

    # Load LogprobsModels
    lp_models = {}
    for model_name in args.models:
        full_name = models_map.get(model_name, model_name)
        print(f"Loading LogprobsModel for {model_name} ({full_name})...")
        lp_models[model_name] = LogprobsModel(full_name, args.device)

    option_labels = ["A", "B", "C", "D"]
    knowledge_flags = {}

    for idx, sample in enumerate(samples):
        knowledge_prompt = _format_mcq_prompt(sample.question, sample.choices)
        flags = {}

        for model_name, lp_model in lp_models.items():
            full_prompt = lp_model.format_chat_prompt(MMLU_SYSTEM_PROMPT_KNOWLEDGE, knowledge_prompt)
            probs = lp_model.get_normalized_probs(full_prompt, option_labels)
            argmax_letter = max(probs, key=probs.get)
            flags[model_name] = argmax_letter

        knowledge_flags[str(idx)] = flags

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx + 1}/{len(samples)}] sample {idx}: {flags}")

    # Save
    with open(args.output, "w") as f:
        json.dump(knowledge_flags, f, indent=2)
    print(f"\nSaved knowledge probes for {len(samples)} samples x {len(args.models)} models to {args.output}")


if __name__ == "__main__":
    main()
