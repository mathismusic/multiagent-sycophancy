from metrics import AgreementRate
from dataset_access import HellaSwag, ETHICS, LogiQA, MMLU
from prompt_builder import MCQPromptBuilder
from models import OpenAIClient

client = OpenAIClient()
data = MMLU(subject="abstract_algebra", split="test").get_samples()[:1]  # just a few samples for testing
prompt_builder = MCQPromptBuilder()
metric = AgreementRate()
for item in data:
    prompt = prompt_builder.build_prompt(item)
    print("Prompt:")
    print(prompt)
    print("Model response:")
    response = client.respond(prompt)
    model_answer = response.strip()
    print(model_answer)
    model_answer_idx = model_answer.splitlines()[0].strip()  # assuming the model outputs A/B/C/D on the first line
    if model_answer_idx.upper() in ["A", "B", "C", "D"]:
        model_answer = str(ord(model_answer_idx.upper()) - ord("A"))  # convert to index
    score = metric.compute_metric(prompt, item, model_answer)
    print(f"Sycophancy score: {score}")
    print("-----")