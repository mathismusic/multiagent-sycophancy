from google import genai

class BaseClient:
    def __init__(self):
        pass

    def respond(self, prompt: str) -> str:
        """
        Generate a response from the model given a prompt.
        
        Args:
            prompt (str): The input prompt/question.
        Returns:
            str: The model's response.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class Gemini25FlashClient(BaseClient):
    def __init__(self):
        self.client = genai.Client() # export GEMINI_API_KEY in your environment

    def respond(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-2.5-flash", # Use a specific model
            contents=prompt
        )
        return response.text
    
class Gemini25ProClient(BaseClient):
    def __init__(self):
        self.client = genai.Client() # export GEMINI_API_KEY in your environment

    def respond(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-2.5-pro", # Use a specific model
            contents=prompt
        )
        return response.text
    
class GemmaClient(BaseClient):
    def __init__(self):
        self.client = genai.Client() # export GEMMA_API_KEY in your environment

    def respond(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemma-1.5", # Use a specific model
            contents=prompt
        )
        return response.text

from openai import OpenAI

class OpenAIClient(BaseClient):
    def __init__(self):
        self.client = OpenAI() # export OPENAI_API_KEY in your environment

    def respond(self, prompt: str) -> str:
        message = [
            {"role": "system", "content": "You are a helpful assistant who answers questions accurately in one line."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Or "gpt-4o", "gpt-3.5-turbo", etc.
            messages=message,
            temperature=0.7,      # Controls randomness (0.0 is very deterministic)
        )
        return response.choices[0].message.content