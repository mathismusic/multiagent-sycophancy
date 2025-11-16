import openai
from abstract_model import BaseLLMDebater
from openai.error import (
    RateLimitError, APIError, ServiceUnavailableError,
    APIConnectionError, InvalidRequestError
)

class GPTDebater(BaseLLMDebater):
    def __init__(self, model_name: str = "gpt-3.5-turbo", dataset: str = "SQA"):
        super().__init__(name="gpt3", dataset=dataset)
        self.model_name = model_name

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, ValueError),
        max_tries=5
    )
    def gen_ans(self, sample, convincing_samples=None,
                additional_instruc=None, intervene=False) -> dict:

        contexts = prepare_context_for_chat_assistant(sample, convincing_samples, intervene, self.dataset)

        if additional_instruc:
            # last message is user in your helper; append extra instructions
            contexts[-1]['content'] += " ".join(additional_instruc)

        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=contexts
        )

        output = completion['choices'][0]['message']['content']

        if not output or "{" not in output or "}" not in output:
            raise ValueError("cannot find { or } in the model output.")

        result = parse_json(output)
        if result == "ERR_SYNTAX" or not result:
            raise ValueError("incomplete JSON format.")

        return self._normalize_answer(result)
