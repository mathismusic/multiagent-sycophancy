import argparse
import json
import random
import re
import os
from dataclasses import dataclass
from typing import List, Optional, Any as AnyType
from datetime import datetime
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

import adapters
from typing import List, Optional, Literal


# By default looks for a .env file in the current working directory
load_dotenv()
# -----------------------------
# Model configurations
# -----------------------------

models_map = {
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3b": "Qwen/Qwen-3B-Instruct",
    "qwen7b": "Qwen/Qwen-7B-Instruct",
    "qwen14b": "Qwen/Qwen-14B-Instruct",
}


## weite  

models_to_backends = {"llama3b":"hf",
                      "llama8b":"hf",
                      "llama1b":"hf",
                        "qwen3b":"hf",
                        "qwen7b":"hf",
                        "qwen14b":"hf",
                      }

# -----------------------------
@dataclass
class GenConfig:
    max_new_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42
    course_name: str = "syco"  # for UIUC adapter


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

def set_seed(seed: int = GenConfig.seed):
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


# -----------------------------
# MMLU loading (validation)
# -----------------------------
def load_mmlu_from_hf(
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
    ds = load_dataset("cais/mmlu", "all", split=hf_split)

    samples: List[MMLUSample] = []
    for ex in ds:
        question = ex["question"]
        choices = list(ex["choices"])
        correct_idx = int(ex["answer"])
        subject = ex["subject"]
        samples.append(MMLUSample(question=question, choices=choices, correct_idx=correct_idx, subject=subject))

    # Ensure randomization (uses global RNG; seed with set_seed() if you want reproducibility)
    random.shuffle(samples)

    if max_items is not None:
        samples = samples[:max_items]

    print(f"[data] loaded {len(samples)} MMLU {split} samples")
    return samples


def build_textgen_pipeline(model, device: str):
    model_name = models_map[model]
    print(f"[load] building pipeline for {model_name} on device {device} ...")

    backend = models_to_backends[model]
    match backend:
        case "hf":
            # hf_token = os.environ.get("HF_TOKEN")
            hf_token = os.getenv("HF_TOKEN")
            model_id = model_name
            tok = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                token=hf_token,
            )
            if device == "cuda":
                dtype = torch.bfloat16
            elif device == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32

            model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token,
            )
            if device == "cuda":
                model = model.to("cuda")
                pipe_device = 0
            elif device == "mps":
                model = model.to("mps")
                pipe_device = -1
            else:
                pipe_device = -1

            pipe = pipeline("text-generation", model=model, tokenizer=tok, device=pipe_device, pad_token_id=tok.eos_token_id)

        case "gemini":
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            pipe = adapters.GeminiAdapter(api_key=gemini_api_key, model_name=model_name)
        case "uiuc":
            uc_key = os.environ.get("UC_KEY")
            pipe = adapters.UIUCAdapter(
                model_name=model_name,
                uc_key=uc_key,
                course_name="CS-546",
            )
    print("[load] pipeline ready")
    return pipe

def chat(pipe, system: str, user: str, gen: GenConfig) -> str:
    prompt = (
        f"<|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{user}\n<|end|>\n<|assistant|>\n"
    )
    out = pipe(
        prompt,
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature,
        top_p=gen.top_p,
        repetition_penalty=gen.repetition_penalty,
        do_sample=(gen.temperature > 0),
    )[0]["generated_text"]
    if "<|assistant|>" in out:
        return out.split("<|assistant|>", 1)[-1].strip()
    return out

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
    m = re.search(r"\b([AaBb])[\.\)\:]", t)
    if m:
        return m.group(1).upper()

    return None

## for mirroing
def parse_yes_no(text: str) -> Optional[str]:
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

## for who said and delusion
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

## picking sides
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

mmlu_metrics_map = {
    "picking_sides": mmlu_to_pickside_items,
    "mirroring": mmlu_to_mirroring_items,
    "who_said": mmlu_to_whosaid_items,
    "delusion": mmlu_to_delusion_items,
}

parsing_metrics_map = {
    "picking_sides": parse_me_friend,
    "mirroring": parse_yes_no,
    "who_said": parse_correct_incorrect,
    "delusion": parse_correct_incorrect,
}