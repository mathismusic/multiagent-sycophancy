import os
import torch
import backoff
from transformers import AutoModelForCausalLM, AutoTokenizer
from abstract_model import BaseLLMDebater
from utils import prepare_context_for_bard, parse_json, invalid_result


class LlamaDebater(BaseLLMDebater):
    """
    Generic debater using Hugging Face Meta-Llama Instruct models.

    You can instantiate it with:
      - "meta-llama/Llama-3.1-8B-Instruct"
      - "meta-llama/Llama-3.2-3B-Instruct"
      - "meta-llama/Llama-3.2-1B-Instruct"
    """

    def __init__(
        self,
        name: str,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        dataset: str = "SQA",
        device: str | None = None,
        max_new_tokens: int = 512,
    ):
        super().__init__(name=name, dataset=dataset)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # --- Device selection: prefer MPS on macOS, then CUDA, then CPU ---
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Choose dtype based on device
        if self.device in ("cuda", "mps"):
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

        # For mps/cpu, move explicitly
        if self.device != "cuda":
            self.model.to(self.device)

    @backoff.on_exception(
        backoff.expo,
        (RuntimeError, ValueError, TypeError),
        max_tries=5
    )
    def gen_ans(
        self,
        sample,
        convincing_samples=None,
        additional_instruc=None,
        intervene: bool = False,
    ) -> dict:
        """
        Mirrors BardDebater.gen_ans, but uses a local HF Llama model.
        """

        # Reuse your existing context builder for Bard-style formatting
        msg, cs, us = prepare_context_for_bard(sample, convincing_samples, intervene, self.dataset)

        # Flatten context into a single text string
        context_text = ""
        for c in cs + us:
            if isinstance(c, tuple):
                context_text += f"User: {c[0]}\nAssistant: {c[1]}\n"
            else:
                context_text += str(c) + "\n"

        if additional_instruc:
            context_text += " ".join(additional_instruc)

        # Build a chat-style prompt for Llama 3 using the chat template
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful reasoning assistant. "
                    "You MUST answer strictly in JSON format: "
                    '{"reasoning": "", "answer": "", "confidence_level": ""}.'
                ),
            },
            {"role": "user", "content": context_text},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Cut off the prompt part, keep only generated continuation
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        if not generated_text:
            raise ValueError("Empty response from Llama")

        result = parse_json(generated_text)
        if result == "ERR_SYNTAX" or not result:
            raise ValueError("incomplete JSON format.")

        return self._normalize_answer(result)
