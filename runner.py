from metrics import AgreementRate
from dataset_access import HellaSwag, ETHICS, LogiQA, MMLU
from prompt_builder import MCQPromptBuilder
from models import Gemini25FlashClient
import json

client = Gemini25FlashClient()
data = MMLU(subject="abstract_algebra", split="test").get_samples()[:1]  # just a few samples for testing

data = HellaSwag(split="validation").get_samples()[:1]  # just a few samples for testing
prompt_builder = MCQPromptBuilder()
metric = AgreementRate()
for item in data:
    prompt = prompt_builder.build_prompt(item)
    print("Prompt:")
    print(prompt)
    print("Model response:")
    response = client.respond(prompt)
    model_answer = response
    print(model_answer)
    model_answer_json = json.loads(model_answer)
    model_answer_idx = ord(model_answer_json["answer"]) - ord('A')

    score = metric.compute_metric(prompt, item, model_answer_idx)
    print(f"Sycophancy score: {score}")
    print("-----")
