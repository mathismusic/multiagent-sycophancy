import os
import backoff
from curl_cffi import CurlError
from requests.exceptions import RequestException
from anthropic import Anthropic
from json import JSONDecodeError
from abstract_model import BaseLLMDebater

class ClaudeDebater(BaseLLMDebater):
    def __init__(self, dataset: str = "SQA"):
        super().__init__(name="claude", dataset=dataset)
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @backoff.on_exception(backoff.expo, (CurlError, RequestException, ValueError), max_tries=3)
    def gen_ans(self, sample, convincing_samples=None,
                additional_instruc=None, intervene=False) -> dict:
        # You already have this helper:
        contexts = prepare_context(sample, convincing_samples, intervene, self.dataset)

        if additional_instruc:
            # For Claude messages API, everything as a single user message is fine
            contexts += " ".join(additional_instruc)

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",  # your chosen model
            max_tokens=1000,
            messages=[{"role": "user", "content": contexts}]
        )

        output = response.content[0].text

        result = parse_json(output)
        if result == "ERR_SYNTAX" or not result:
            result = invalid_result(self.dataset)

        return self._normalize_answer(result)
