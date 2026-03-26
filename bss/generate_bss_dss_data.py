"""
Generate the two disjoint MMLU datasets used by the pipeline:
  - data_for_bss.csv   (used by bss_calc.py to compute base sycophancy scores)
  - data_for_dss.csv   (used by multiagent-debate.py for discussions)

Both are drawn from the MMLU *test* split, shuffled with a fixed seed,
then split per subject so there is zero overlap.

Arguments:
  --subjects          MMLU subjects to include (default: elementary_mathematics,
                      professional_law, machine_learning, business_ethics,
                      high_school_biology)
  --bss_per_subject   Number of samples per subject for BSS score computation
                      (default: 20)
  --dss_per_subject   Number of samples per subject for multi-agent discussions
                      (default: 4)
  --seed              Random seed for shuffling (default: 42)
  --bss_out           Output CSV path for BSS data (default: data_for_bss.csv)
  --dss_out           Output CSV path for discussion data (default: data_for_dss.csv)

Usage:
  # Generate with defaults (20 BSS + 4 discussion samples per subject)
  python generate_bss_dss_data.py

  # Generate 50 samples per subject for both BSS and discussions
  python generate_bss_dss_data.py --bss_per_subject 50 --dss_per_subject 50
"""

import argparse
import random

from utils import load_mmlu_from_hf, mmlu_list_to_csv, count_by_subject
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Generate BSS and DSS datasets from MMLU test split")
    parser.add_argument("--subjects", nargs="+", default=[
        "elementary_mathematics",
        "professional_law",
        "machine_learning",
        "business_ethics",
        "high_school_biology",
    ])
    parser.add_argument("--bss_per_subject", type=int, default=20,
                        help="Number of samples per subject for BSS")
    parser.add_argument("--dss_per_subject", type=int, default=4,
                        help="Number of samples per subject for discussions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bss_out", type=str, default="data_for_bss.csv")
    parser.add_argument("--dss_out", type=str, default="data_for_dss.csv")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all MMLU test samples for the chosen subjects
    all_samples = load_mmlu_from_hf(args.subjects, split="test")

    # Group by subject
    by_subject = defaultdict(list)
    for s in all_samples:
        by_subject[s.subject].append(s)

    # Shuffle within each subject (seed already set)
    for subj in by_subject:
        random.shuffle(by_subject[subj])

    needed = args.bss_per_subject + args.dss_per_subject
    for subj in args.subjects:
        avail = len(by_subject[subj])
        if avail < needed:
            raise ValueError(
                f"Subject '{subj}' has only {avail} samples but "
                f"{needed} requested ({args.bss_per_subject} BSS + {args.dss_per_subject} DSS)"
            )

    bss_samples = []
    dss_samples = []
    for subj in args.subjects:
        pool = by_subject[subj]
        dss_samples.extend(pool[: args.dss_per_subject])
        bss_samples.extend(pool[args.dss_per_subject : args.dss_per_subject + args.bss_per_subject])

    print(f"BSS samples: {len(bss_samples)}  {count_by_subject(bss_samples)}")
    print(f"DSS samples: {len(dss_samples)}  {count_by_subject(dss_samples)}")

    mmlu_list_to_csv(bss_samples, args.bss_out)
    mmlu_list_to_csv(dss_samples, args.dss_out)

    print(f"\nSaved {args.bss_out} and {args.dss_out}")


if __name__ == "__main__":
    main()
