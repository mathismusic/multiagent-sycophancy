"""
Generate 50 new BSS samples per subject that are disjoint from the 250 DSS
samples your friend already used (50 DSS/subject from MMLU test).

How it works:
  1. Replays the exact same load_split_save_dataset logic your friend ran
     (no_of_debate_samples=250, bss_samples=50, dataset_type=test, seed=42)
     to identify the 50 debate (DSS) samples per subject.
  2. Saves those 250 DSS samples to data_for_dss_prev.csv.
  3. Loads the full MMLU test set, subtracts the 250 DSS samples.
  4. Samples 50 per subject from the remainder for new BSS data.
  5. Saves to data_for_bss_prev.csv.

Usage:
  python generate_bss_data_tmp.py
  python generate_bss_data_tmp.py --new_bss_per_subject 50
"""

import argparse
import random
from collections import defaultdict

from utils import (
    load_mmlu_from_hf,
    mmlu_list_to_csv,
    count_by_subject,
    set_seed,
)


SUBJECTS = [
    "elementary_mathematics",
    "professional_law",
    "machine_learning",
    "business_ethics",
    "high_school_biology",
]


def replay_friends_split(subjects):
    """
    Reproduce the exact split your friend got when running multiagent-debate.py
    with --no_of_debate_samples 250 --bss_samples 50 --dataset_type test.

    Returns (debate_samples, bss_samples) as two lists of MMLUSample.
    """
    # multiagent-debate.py calls set_seed(42) before load_split_save_dataset
    set_seed(42)

    # load_mmlu_from_hf shuffles with the current random state
    all_samples = load_mmlu_from_hf(subjects, split="test")

    # Replay load_split_save_dataset logic
    num_subjects = len(subjects)
    no_of_debate_samples = 250  # friend's value
    bss_samples_count = 50      # friend's value

    requested_per_subject = no_of_debate_samples // num_subjects  # 50
    bss_per_subject = bss_samples_count // num_subjects           # 10
    cap = requested_per_subject + bss_per_subject                 # 60

    # Group by subject with cap
    by_subject = defaultdict(list)
    for sample in all_samples:
        if len(by_subject[sample.subject]) >= cap:
            continue
        by_subject[sample.subject].append(sample)

    # load_split_save_dataset resets seed to 42 and shuffles per subject
    random.seed(42)
    debate_samples = []
    bss_samples = []
    for subj in subjects:
        samples_for_subj = by_subject[subj]
        random.shuffle(samples_for_subj)
        per_subject_for_me = min(requested_per_subject,
                                 min(len(by_subject[s]) for s in subjects))
        debate_samples.extend(samples_for_subj[:per_subject_for_me])
        bss_samples.extend(samples_for_subj[per_subject_for_me:])

    print(f"\nFriend's debate (DSS) samples: {len(debate_samples)}  "
          f"{count_by_subject(debate_samples)}")
    print(f"Friend's BSS samples: {len(bss_samples)}  "
          f"{count_by_subject(bss_samples)}")
    return debate_samples, bss_samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate new BSS data disjoint from friend's DSS samples")
    parser.add_argument("--new_bss_per_subject", type=int, default=50,
                        help="New BSS samples per subject (default: 50)")
    parser.add_argument("--seed", type=int, default=123,
                        help="Seed for sampling new data (default: 123)")
    parser.add_argument("--dss_out", type=str, default="data_for_dss_prev.csv",
                        help="Output CSV for friend's DSS samples (default: data_for_dss_prev.csv)")
    parser.add_argument("--bss_out", type=str, default="data_for_bss_prev.csv",
                        help="Output CSV for new BSS samples (default: data_for_bss_prev.csv)")
    args = parser.parse_args()

    # Step 1: Replay friend's split
    debate_samples, _ = replay_friends_split(SUBJECTS)

    # Step 2: Save the 250 DSS samples
    mmlu_list_to_csv(debate_samples, args.dss_out)
    print(f"\nSaved friend's {len(debate_samples)} DSS samples to {args.dss_out}")

    # Step 3: Load full MMLU test set
    set_seed(0)
    full_samples = load_mmlu_from_hf(SUBJECTS, split="test")

    # Build exclusion set from DSS samples only
    dss_keys = set((s.question, s.subject) for s in debate_samples)

    # Step 4: Subtract DSS samples
    by_subject_remaining = defaultdict(list)
    for s in full_samples:
        if (s.question, s.subject) not in dss_keys:
            by_subject_remaining[s.subject].append(s)

    print("\nRemaining per subject after removing 250 DSS samples:")
    for subj in SUBJECTS:
        print(f"  {subj}: {len(by_subject_remaining[subj])}")

    # Step 5: Sample new BSS data from remainder
    random.seed(args.seed)
    new_bss = []
    for subj in SUBJECTS:
        pool = by_subject_remaining[subj]
        random.shuffle(pool)
        if len(pool) < args.new_bss_per_subject:
            raise ValueError(
                f"Subject '{subj}' has only {len(pool)} remaining samples "
                f"but {args.new_bss_per_subject} requested")
        new_bss.extend(pool[:args.new_bss_per_subject])

    print(f"\nNew BSS samples: {len(new_bss)}  {count_by_subject(new_bss)}")

    # Step 6: Save
    mmlu_list_to_csv(new_bss, args.bss_out)
    print(f"Saved to {args.bss_out}")


if __name__ == "__main__":
    main()
