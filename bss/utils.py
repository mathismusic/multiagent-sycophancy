"""
Shared utility functions for sycophancy scoring and multi-agent debate.
Contains dataset loading, model I/O helpers, parsing functions, and data types.
"""

import random
import re
import os
from dataclasses import dataclass
from typing import List, Optional, Literal
from dotenv import load_dotenv
from collections import defaultdict

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

from prompt import (
    MMLU_USER_PROMPT_MCQ
)

# By default looks for a .env file in the current working directory
load_dotenv()

# -----------------------------
# Model configurations
# -----------------------------

models_map = {
    # "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    # "mistral3b": "mistralai/Ministral-3-3B-Instruct-2512",
    # "mistral8b": "mistralai/Ministral-3-8B-Instruct-2512",
    # "mistral14b": "mistralai/Ministral-3-14B-Instruct-2512",
}

models_to_backends = {
    "llama3b": "hf",
    "llama8b": "hf",
    "llama1b": "hf",
    "qwen3b": "hf",
    "qwen7b": "hf",
    "qwen14b": "hf",
    "qwen32b": "hf",
    "mistral3b": "hf",
    "mistral8b": "hf",
    "mistral14b": "hf",
}

# -----------------------------
# Config
# -----------------------------
@dataclass
class GenConfig:
    max_new_tokens: int = 10
    temperature: float = 0.0   # deterministic
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42
    course_name: str = "syco"


# -----------------------------
# Core data types
# -----------------------------
@dataclass
class MMLUSample:
    question: str
    choices: List[str]
    correct_idx: int
    subject: str


# Metric-specific item types
@dataclass
class PickingSidesItem:
    question: str
    correct: str
    wrong: str
    subject: str


@dataclass
class MirroringItem:
    question: str
    correct: str
    wrong: str
    subject: str


@dataclass
class WhoSaidItem:
    question: str
    correct: str
    wrong: str
    subject: str


@dataclass
class DelusionItem:
    question: str
    correct: str
    wrong: str
    subject: str


@dataclass
class UserStanceItem:
    question: str
    correct: str   # optional but useful for logging
    wrong: str
    subject: str


# -----------------------------
# Utility: seeding & device
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str = "auto") -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------
# Model I/O helpers
# -----------------------------
def build_textgen_pipeline(model, device: str):
    """
    Build a text generation pipeline for the specified model.
    
    Args:
        model: Either a short model name (e.g., 'llama3b') that maps to a full HF model ID,
               or a full HF model ID string.
        device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
    """
    # Check if it's a short name or full model ID
    if model in models_map:
        model_name = models_map[model]
        backend = models_to_backends[model]
    else:
        # Assume it's a full model ID
        model_name = model
        backend = "hf"
    
    print(f"[load] building pipeline for {model_name} on device {device} ...")
    
    match backend:
        case "hf":
            hf_token = os.getenv("HF_TOKEN")
            model_id = model_name
            tok = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                token=hf_token,
            )
            
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
                token=hf_token,
            )

            # With device_map="auto", the model is already on GPU, use device=None for pipeline
            pipe = pipeline("text-generation", model=mdl, tokenizer=tok, pad_token_id=tok.eos_token_id)
    
    print("[load] pipeline ready")
    return pipe


# Toggle to switch between native chat templates and the legacy Phi-style template.
# Set to False to revert to the old hardcoded format.
USE_NATIVE_CHAT_TEMPLATE = True


def _format_prompt_legacy(system: str, user: str) -> str:
    """Legacy Phi-style chat template (hardcoded)."""
    return (
        f"<|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{user}\n<|end|>\n<|assistant|>\n"
    )


def _format_prompt_native(tokenizer, system: str, user: str) -> str:
    """Use the tokenizer's built-in chat template (model-specific)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _extract_reply(text: str, tokenizer=None) -> str:
    """Extract the assistant's reply from generated text."""
    if USE_NATIVE_CHAT_TEMPLATE and tokenizer is not None:
        # For native templates, the prompt is included in the output.
        # Split on common assistant markers.
        # Llama 3: <|start_header_id|>assistant<|end_header_id|>\n\n
        # Qwen 2.5: <|im_start|>assistant\n
        for marker in [
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|im_start|>assistant\n",
            "<|assistant|>\n",
        ]:
            if marker in text:
                return text.rsplit(marker, 1)[-1].strip()
    # Legacy fallback
    if "<|assistant|>" in text:
        return text.split("<|assistant|>", 1)[-1].strip()
    return text.strip()


def chat(pipe, system: str, user: str, gen: GenConfig) -> str:
    """
    Single-turn chat with a model pipeline.
    """
    if USE_NATIVE_CHAT_TEMPLATE:
        prompt = _format_prompt_native(pipe.tokenizer, system, user)
    else:
        prompt = _format_prompt_legacy(system, user)

    out = pipe(
        prompt,
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature,
        top_p=gen.top_p,
        repetition_penalty=gen.repetition_penalty,
        do_sample=(gen.temperature > 0),
    )[0]["generated_text"]
    return _extract_reply(out, pipe.tokenizer if USE_NATIVE_CHAT_TEMPLATE else None)


def chat_batch(pipe, system: str, users: List[str], gen: GenConfig, batch_size: int = 8) -> List[str]:
    """
    Batched chat inference for multiple user prompts.
    """
    if USE_NATIVE_CHAT_TEMPLATE:
        prompts = [_format_prompt_native(pipe.tokenizer, system, u) for u in users]
    else:
        prompts = [_format_prompt_legacy(system, u) for u in users]

    # Try batch inference (works for HF pipelines)
    try:
        outputs = pipe(
            prompts,
            max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p,
            repetition_penalty=gen.repetition_penalty,
            do_sample=(gen.temperature > 0),
            batch_size=batch_size
        )
        results = []
        tok = pipe.tokenizer if USE_NATIVE_CHAT_TEMPLATE else None
        for out in outputs:
            text = out[0]["generated_text"]
            results.append(_extract_reply(text, tok))
        return results
    except Exception:
        # Fallback for backends that don't support batching (e.g. API adapters)
        return [chat(pipe, system, u, gen) for u in users]


# -----------------------------
# Parsing helpers
# -----------------------------
def parse_AB(text: str) -> Optional[str]:
    """
    Try to extract 'A' or 'B' from the reply.
    """
    if not text:
        return None
    t = text.strip()

    # First alphabetic char
    for ch in t:
        if ch.isalpha():
            ch_low = ch.lower()
            if ch_low == "a":
                return "A"
            if ch_low == "b":
                return "B"
            break

    # Standalone A/B
    m = re.search(r"\b([AaBb])\b", t)
    if m:
        return m.group(1).upper()

    # 'option A' etc.
    m = re.search(r"\boption\s+([AaBb])\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 'A.' 'B)' etc.
    m = re.search(r"\b([AaBb])[\.\):]", t)
    if m:
        return m.group(1).upper()

    return None


def parse_option(text: str) -> Optional[str]:
    """
    Try to extract 'A', 'B', 'C', 'D', 'E', or 'F' from the reply.
    """
    if not text:
        return None
    t = text.strip()
    valid_options = "abcdef"

    # First alphabetic char
    for ch in t:
        if ch.isalpha():
            ch_low = ch.lower()
            if ch_low in valid_options:
                return ch.upper()
            break

    # Standalone A/B/C/D/E/F
    m = re.search(r"\b([A-Fa-f])\b", t)
    if m:
        return m.group(1).upper()

    # 'option A' etc.
    m = re.search(r"\boption\s+([A-Fa-f])\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 'A.' 'B)' 'C:' etc.
    m = re.search(r"\b([A-Fa-f])[\.\):]", t)
    if m:
        return m.group(1).upper()

    return None


def parse_yes_no(text: str) -> Optional[str]:
    """
    Parse 'yes' or 'no' from reply.
    """
    if not text:
        return None
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return None


def parse_correct_incorrect(text: str) -> Optional[str]:
    """
    Extract 'correct' or 'incorrect' from reply.
    'incorrect' checked first because it contains 'correct'.
    """
    if not text:
        return None
    t = text.strip().lower()

    if "incorrect" in t:
        return "incorrect"
    if "not correct" in t or "isn't correct" in t or "is not correct" in t:
        return "incorrect"
    if "correct" in t:
        return "correct"
    return None


def parse_me_friend(text: str) -> Optional[str]:
    """
    Parse 'me' or 'my friend'.
    """
    if not text:
        return None
    t = text.strip().lower()
    # First word heuristic
    if t.startswith("me"):
        return "me"
    if t.startswith("my friend"):
        return "my friend"

    # Search
    if "my friend" in t:
        return "my friend"
    if "me" in t:
        return "me"
    return None


# -----------------------------
# Build items for each metric
# -----------------------------
def pick_random_wrong(choices: List[str], correct_idx: int) -> Optional[str]:
    if len(choices) < 2:
        return None
    wrong_idxs = [i for i in range(len(choices)) if i != correct_idx]
    if not wrong_idxs:
        return None
    return choices[random.choice(wrong_idxs)]


def mmlu_to_pickside_items(samples: List[MMLUSample]) -> List[PickingSidesItem]:
    items = []
    for s in samples:
        wrong = pick_random_wrong(s.choices, s.correct_idx)
        if wrong is None:
            continue
        correct = s.choices[s.correct_idx]
        items.append(PickingSidesItem(question=s.question, correct=correct, wrong=wrong, subject=s.subject))
    print(f"[data] built {len(items)} picking_sides items from MMLU")
    return items


def mmlu_to_mirroring_items(samples: List[MMLUSample]) -> List[MirroringItem]:
    items = []
    for s in samples:
        wrong = pick_random_wrong(s.choices, s.correct_idx)
        if wrong is None:
            continue
        correct = s.choices[s.correct_idx]
        items.append(MirroringItem(question=s.question, correct=correct, wrong=wrong, subject=s.subject))
    print(f"[data] built {len(items)} mirroring items from MMLU")
    return items


def mmlu_to_whosaid_items(samples: List[MMLUSample]) -> List[WhoSaidItem]:
    items = []
    for s in samples:
        wrong = pick_random_wrong(s.choices, s.correct_idx)
        if wrong is None:
            continue
        correct = s.choices[s.correct_idx]
        items.append(WhoSaidItem(question=s.question, correct=correct, wrong=wrong, subject=s.subject))
    print(f"[data] built {len(items)} whosaid items from MMLU")
    return items


def mmlu_to_delusion_items(samples: List[MMLUSample]) -> List[DelusionItem]:
    items = []
    for s in samples:
        wrong = pick_random_wrong(s.choices, s.correct_idx)
        if wrong is None:
            continue
        correct = s.choices[s.correct_idx]
        items.append(DelusionItem(question=s.question, correct=correct, wrong=wrong, subject=s.subject))
    print(f"[data] built {len(items)} delusion items from MMLU")
    return items


def mmlu_to_user_stance_items(samples: List[MMLUSample]) -> List[UserStanceItem]:
    items = []
    for s in samples:
        correct = s.choices[s.correct_idx]
        wrong = pick_random_wrong(s.choices, s.correct_idx)
        if wrong is None:
            continue
        items.append(UserStanceItem(question=s.question, correct=correct, wrong=wrong, subject=s.subject))
    return items


# Mapping from metric names to item builder functions
mmlu_metrics_map = {
    "picking_sides": mmlu_to_pickside_items,
    "mirroring": mmlu_to_mirroring_items,
    "who_said": mmlu_to_whosaid_items,
    "delusion": mmlu_to_delusion_items,
    "agreement_rate": mmlu_to_user_stance_items,
    "sycophant_with_knowledge": mmlu_to_user_stance_items,
    "confident_sycophancy": mmlu_to_user_stance_items,
}

# Mapping from metric names to parsing functions
parsing_metrics_map = {
    "picking_sides": parse_me_friend,
    "mirroring": parse_yes_no,
    "who_said": parse_correct_incorrect,
    "delusion": parse_correct_incorrect,
    "agreement_rate": parse_correct_incorrect,
    "sycophant_with_knowledge": parse_correct_incorrect,
    "confident_sycophancy": parse_correct_incorrect,
}


def count_by_subject(samples):
    c = defaultdict(int)
    for s in samples:
        c[s.subject] += 1
    return dict(c)


# -----------------------------
# MMLU loading
# -----------------------------
def load_mmlu_from_hf(
        subjects,
        split: Literal["train", "test", "val"] = "val",
        max_items: Optional[int] = None,
) -> List[MMLUSample]:
    """
    Load MMLU split from Hugging Face.

    split: "train", "test", or "val"
           (we map "val" -> "validation" for HF)
    """
    # Map our shorthand to the HF split name
    hf_split_map = {
        "train": "train",
        "test": "test",
        "val": "validation",
    }
    hf_split = hf_split_map[split]  # safe: split is constrained by Literal

    print(f"[data] downloading cais/mmlu ('all' / '{hf_split}') from Hugging Face ...")
    ds_all = load_dataset("cais/mmlu", "all", split=hf_split)

    # Filtering out only relevant samples
    ds = ds_all.filter(lambda x: x["subject"] in subjects)

    samples: List[MMLUSample] = []
    for ex in ds:
        question = ex["question"]
        choices = list(ex["choices"])
        correct_idx = int(ex["answer"])
        subject = ex["subject"]
        samples.append(MMLUSample(question=question, choices=choices, correct_idx=correct_idx, subject=subject))

    random.shuffle(samples)

    if max_items is not None:
        samples = samples[:max_items]

    print(f"[data] loaded {len(samples)} MMLU {split} samples")
    return samples


def mmlu_list_to_csv(samples, csv_path: str = "mmlu_debate_samples.csv"):
    """
    Convert a list of MMLUSample-like objects to a CSV.

    Each sample is expected to have:
      - question: str
      - choices: List[str]
      - correct_idx: int
      - subject: str
    """
    answer_labels = ["A", "B", "C", "D"]  # standard MMLU labels

    rows = []
    for s in samples:
        row = {
            "subject": s.subject,
            "question": s.question,
            "answer": s.correct_idx,
            "answer_letter": answer_labels[s.correct_idx],
        }

        # choices -> choice_A, choice_B, ...
        for i, label in enumerate(answer_labels):
            row[f"choice_{label}"] = s.choices[i] if i < len(s.choices) else None

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples to {csv_path}")


def load_mmlu_from_csv(csv_path: str) -> List[MMLUSample]:
    """Load MMLUSample objects from a CSV previously written by mmlu_list_to_csv."""
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        choices = [row[f"choice_{label}"] for label in ["A", "B", "C", "D"]]
        samples.append(MMLUSample(
            question=row["question"],
            choices=choices,
            correct_idx=int(row["answer"]),
            subject=row["subject"],
        ))
    print(f"[data] loaded {len(samples)} MMLU samples from {csv_path}")
    return samples


def load_split_save_dataset(args):
    """
    Load MMLU dataset and split into debate samples and BSS samples.
    
    Args:
        args: Namespace with subjects, dataset_type, no_of_debate_samples, bss_samples (or max_items)
    
    Returns:
        (debate_samples, bss_samples) - two lists of MMLUSample objects
    """
    print(f"\nLoading MMLU samples for subjects: {args.subjects}, dataset_type: {args.dataset_type}")

    mmlu_samples = load_mmlu_from_hf(args.subjects, args.dataset_type)
    print(f"\nLength of MMLU_Samples: ", len(mmlu_samples))

    num_subjects = len(args.subjects)

    # ---------------------------------------------------
    # How many total samples do YOU want?
    # ---------------------------------------------------
    requested_total = args.no_of_debate_samples
    requested_per_subject = requested_total // num_subjects
    
    # Support both bss_samples and max_items attribute names
    bss_total = getattr(args, 'bss_samples', None) or getattr(args, 'max_items', 0)
    bss_requested_per_subject = bss_total // num_subjects

    # Group by subject
    by_subject = defaultdict(list)
    for sample in mmlu_samples:
        if len(by_subject[sample.subject]) >= bss_requested_per_subject + requested_per_subject:
            continue
        by_subject[sample.subject].append(sample)

    print("Counts per subject:")
    for subj, lst in by_subject.items():
        print(subj, len(lst))

    # Max we can actually take per subject (bounded by available data)
    max_per_subject_possible = min(len(by_subject[subj]) for subj in args.subjects)

    # Final per-subject count we will use (cannot exceed available)
    per_subject_for_me = min(requested_per_subject, max_per_subject_possible)

    print(f"\nRequested total: {requested_total}")
    print(f"Subjects: {num_subjects}")
    print(f"Requested per subject: {requested_per_subject}")
    print(f"Max possible per subject (from data): {max_per_subject_possible}")
    print(f"Using per_subject_for_me = {per_subject_for_me}")

    # ---------------------------------------------------
    # Build splits: mine vs others
    # ---------------------------------------------------
    debate_samples = []
    bss_samples = []

    random.seed(42)

    for subj in args.subjects:
        samples_for_subj = by_subject[subj]
        random.shuffle(samples_for_subj)  # randomize within subject

        # Take first N for you
        mine_subj = samples_for_subj[:per_subject_for_me]
        # Everything else from this subject goes to others
        others_subj = samples_for_subj[per_subject_for_me:]

        debate_samples.extend(mine_subj)
        bss_samples.extend(others_subj)

    print(f"\nMy samples total: {len(debate_samples)}")
    print(f"Other samples total: {len(bss_samples)}")

    print("\nMy samples per subject:", count_by_subject(debate_samples))
    print("Other samples per subject:", count_by_subject(bss_samples))

    print("\nSaving debate and bss samples to CSV...")
    mmlu_list_to_csv(debate_samples, "mmlu_debate_samples.csv")
    mmlu_list_to_csv(bss_samples, "mmlu_bss_samples.csv")

    return debate_samples, bss_samples

def _format_mcq_prompt(question: str, choices: List[str]) -> str:
    """
    Format the original unchanged question as a multiple-choice prompt.
    """
    labels = ["A", "B", "C", "D", "E", "F"]
    lines = []
    for i, ch in enumerate(choices):
        lab = labels[i] if i < len(labels) else str(i)
        lines.append(f"{lab}. {ch}")
    choices_block = "\n".join(lines)
    return MMLU_USER_PROMPT_MCQ.format(question=question, choices_block=choices_block)


def _correct_letter_for_sample(s: MMLUSample) -> str:
    labels = ["A", "B", "C", "D", "E", "F"]
    if 0 <= s.correct_idx < len(labels):
        return labels[s.correct_idx]
    return str(s.correct_idx)
