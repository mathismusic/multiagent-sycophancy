"""
LogprobsModel: A wrapper around HuggingFace models that computes logprobs for specific 
output token sequences (like "correct", "incorrect", "yes", "no", etc.)

Used for compute_confident_sycophancy metric.
"""

import os
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import GenConfig


class LogprobsModel:
    """
    A wrapper around HuggingFace models that computes logprobs for specific 
    output token sequences (like "correct", "incorrect", "yes", "no", etc.)
    
    Uses the SAME chat format as bss_calc.py's chat() function:
      <|system|>\n{system}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        hf_token = os.getenv("HF_TOKEN")
        
        print(f"[LogprobsModel] Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True, 
            token=hf_token,
        )
        
        print(f"[LogprobsModel] Loading model {model_name}...")
        # MPS (Apple Silicon) doesn't support BFloat16, use float32
        # Check if we're using MPS
        use_mps = device == "mps" or (device == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if use_mps:
            dtype = torch.float32  # MPS doesn't support bfloat16
            # Load to CPU first, then move to MPS to avoid offloading issues
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,  # Don't use auto offloading
                torch_dtype=dtype,
                token=hf_token,
            )
            self.model = self.model.to("mps")
            print(f"[LogprobsModel] Using float32 and loaded to MPS for compatibility")
        else:
            dtype = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=dtype,
                token=hf_token,
            )
        self.model.eval()
        
        # Find the device the model is on
        self.device = next(self.model.parameters()).device
        print(f"[LogprobsModel] Model loaded on device: {self.device}")
    
    def format_chat_prompt(self, system: str, user: str) -> str:
        """
        Format as chat prompt, consistent with utils.py's chat() function.

        Uses the tokenizer's native chat template when USE_NATIVE_CHAT_TEMPLATE
        is True (default), otherwise falls back to the legacy Phi-style format.
        """
        from utils import USE_NATIVE_CHAT_TEMPLATE

        if USE_NATIVE_CHAT_TEMPLATE:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            return (
                f"<|system|>\n{system}\n<|end|>\n"
                f"<|user|>\n{user}\n<|end|>\n<|assistant|>\n"
            )
    
    def get_token_logprobs(
        self, 
        prompt: str, 
        target_sequences: List[str]
    ) -> Dict[str, float]:
        """
        Compute the log-probability of each target sequence being generated
        as the next tokens after the prompt.
        
        Args:
            prompt: The input prompt (already formatted with format_chat_prompt)
            target_sequences: List of possible completions (e.g., ["correct", "incorrect"])
        
        Returns:
            Dict mapping each target sequence to its log probability
        """
        # Tokenize the prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        results = {}
        
        with torch.no_grad():
            for target in target_sequences:
                # Tokenize the target sequence
                target_ids = self.tokenizer.encode(target, add_special_tokens=False)
                
                # Concatenate prompt + target
                full_ids = torch.cat([
                    prompt_ids, 
                    torch.tensor([target_ids], device=self.device)
                ], dim=1)
                
                # Get model outputs with logits
                outputs = self.model(full_ids)
                logits = outputs.logits  # shape: (1, seq_len, vocab_size)
                
                # Compute log probabilities for each target token
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Sum log probabilities for target tokens
                # logits[0, i] predicts token at position i+1
                total_logprob = 0.0
                prompt_len = prompt_ids.shape[1]
                
                for i, tok_id in enumerate(target_ids):
                    logit_pos = prompt_len - 1 + i
                    token_logprob = log_probs[0, logit_pos, tok_id].item()
                    total_logprob += token_logprob
                
                results[target] = total_logprob
        
        return results
    
    def get_normalized_probs(
        self, 
        prompt: str, 
        target_sequences: List[str]
    ) -> Dict[str, float]:
        """
        Get probabilities normalized to sum to 1 across the target sequences.
        """
        logprobs = self.get_token_logprobs(prompt, target_sequences)
        
        # Convert to probabilities and normalize
        probs = {k: torch.exp(torch.tensor(v)).item() for k, v in logprobs.items()}
        total = sum(probs.values())
        
        if total > 0:
            normalized = {k: v / total for k, v in probs.items()}
        else:
            normalized = {k: 1.0 / len(target_sequences) for k in target_sequences}
        
        return normalized
    
    def generate(self, prompt: str, gen_config: GenConfig) -> str:
        """Generate text (for compatibility with existing code)."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature if gen_config.temperature > 0 else 0.0,
                top_p=gen_config.top_p,
                repetition_penalty=gen_config.repetition_penalty,
                do_sample=(gen_config.temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return generated.strip()
