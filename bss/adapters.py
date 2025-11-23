"""Adapters to provide a pipeline-like interface for different backends.

Each adapter implements __call__(inputs, max_new_tokens=..., temperature=..., **kwargs)
and returns a list of dicts like the Hugging Face pipeline: [{"generated_text": text}].

This file contains:
 - APIAdapterBase: small base class
 - GeminiAdapter: adapter for Google Gemini API
 - UIUCChatAdapter: adapter for opensource caht models (kindly provided by UIUC)
 - build_model_pipe: factory for convenience
"""
import re
import requests
import google.generativeai as genai


class APIAdapterBase:
    def __call__(self, inputs: str, max_new_tokens: int = 512, temperature: float = 0.0, **kwargs) -> list[dict[str, str]]:
        text = self.call_api(inputs, {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            **kwargs,
        })
        return [{"generated_text": text}]

    def call_api(self, prompt: str, params: dict) -> str:
        raise NotImplementedError()

class GeminiAdapter(APIAdapterBase):
    def __init__(self, api_key, model_name):
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def call_api(self, prompt: str, params: dict):
        # Extract system and user content directly, leveraging the fixed prompt structure
        system_match = re.search(r'<\|system\|>\n(.*?)\n<\|end\|>', prompt, re.DOTALL)
        user_match = re.search(r'<\|user\|>\n(.*?)\n<\|end\|>', prompt, re.DOTALL)

        system_content = system_match.group(1).strip() if system_match else ""
        user_content = user_match.group(1).strip() if user_match else ""

        combined_content = ""
        if system_content:
            combined_content += system_content + "\n"
        if user_content:
            combined_content += user_content

        if not combined_content.strip():
            raise ValueError("Could not parse prompt into valid Gemini conversation history.")

        # Gemini's generate_content expects a list of Contents, where system/user roles are typically combined into a 'user' role.
        gemini_contents = [{
            "role": "user",
            "parts": [{"text": combined_content.strip()}]
        }]
        print(gemini_contents)

        generation_config = {
            "max_output_tokens": params.get("max_new_tokens", 512),
            "temperature": params.get("temperature", 0.0),
            "top_p": params.get("top_p", 1.0),
        }

        # Only include top_p if temperature is > 0, otherwise it might conflict or be redundant
        if generation_config["temperature"] == 0.0:
            generation_config.pop("top_p", None)

        response = self.model.generate_content(
            contents=gemini_contents,
            generation_config=generation_config,
            stream=False
        )
        try:
            return response.text
        except ValueError as e:
            print(f"ValueError when accessing response.text: {e}")
            print("Prompt Feedback:", response.prompt_feedback)
            if response.candidates:
                print("Candidate 0 Finish Reason:", response.candidates[0].finish_reason)
                print("Candidate 0 Safety Ratings:", response.candidates[0].safety_ratings)
            else:
                print("No candidates returned.")
            raise # Re-raise the original error after printing debug info

class UIUCAdapter(APIAdapterBase):
    UIUC_CHAT_URL = "https://uiuc.chat/api/chat-api/chat"
    def __init__(self, uc_key, model_name, course_name):
        self.model_name = model_name
        self.api_key = uc_key
        self.course_name = course_name

    def call_api(self, prompt: str, params: dict):
        headers = {"Content-Type": "application/json"}
        pattern = re.compile(r'<\|(.*?)\|>\s*(.*?)\s*<\|end\|>', re.DOTALL)

        messages = []
        for match in pattern.finditer(prompt):
            role, content = match.groups()
            messages.append({
                "role": role,
                "content": content.strip()
            })
        payload = {
        "model": self.model_name,
        "messages": messages,
        "api_key": self.api_key,
        "course_name": self.course_name,
        "stream": False,
        "temperature": params.get("temperature", 0.0),
        "max_tokens": params.get("max_new_tokens", 512),
        "retrieval_only": False,
    }

        print(headers)
        print(payload)
        resp = requests.post(self.UIUC_CHAT_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        text = eval(resp.text.strip())["message"]
        return text
