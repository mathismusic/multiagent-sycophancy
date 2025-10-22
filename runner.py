from metrics import AgreementRate
from dataset_access import HellaSwag, ETHICS, LogiQA, MMLU
from prompt_builder import MCQPromptBuilder

import os

# run gemini (take api key as env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# use google ai studio gemini client
from google import genai

client = genai.Client() # will automatically pick up the GEMINI_API_KEY environment variable

data = MMLU(subject="abstract_algebra", split="test").get_samples()[:5]  # just a few samples for testing
prompt_builder = MCQPromptBuilder()
metric = AgreementRate()
for item in data:
    prompt = prompt_builder.build_prompt(item)
    print("Prompt:")
    print(prompt)
    print("Model response:")
    response = client.models.generate_content(
        model="gemini-2.5-flash", # Use a specific model
        contents=prompt
    )
    model_answer = response.text.strip()
    print(model_answer)
    model_answer_idx = model_answer.splitlines()[0].strip()  # assuming the model outputs A/B/C/D on the first line
    if model_answer_idx.upper() in ["A", "B", "C", "D"]:
        model_answer = str(ord(model_answer_idx.upper()) - ord("A"))  # convert to index
    score = metric.compute_metric(prompt, item, model_answer)
    print(f"Sycophancy score: {score}")
    print("-----")