import google.generativeai as genai
from google.api_core.exceptions import ServiceUnavailable
from abstract_model import BaseLLMDebater

class BardDebater(BaseLLMDebater):
    def __init__(self, model_name: str = "gemini-2.5-flash", dataset: str = "SQA"):
        super().__init__(name="bard", dataset=dataset)
        self.model_name = model_name
        genai.configure(api_key=os.environ["PALM_API_KEY"])
        self.model = genai.GenerativeModel(self.model_name)

    @backoff.on_exception(
        backoff.expo,
        (ServiceUnavailable, ValueError, TypeError),
        max_tries=5
    )
    def gen_ans(self, sample, convincing_samples=None,
                additional_instruc=None, intervene=False) -> dict:

        msg, cs, us = prepare_context_for_bard(sample, convincing_samples, intervene, self.dataset)

        context_text = ""
        for c in cs + us:
            if isinstance(c, tuple):
                context_text += f"User: {c[0]}\nAssistant: {c[1]}\n"
            else:
                context_text += str(c) + "\n"

        if additional_instruc:
            context_text += " ".join(additional_instruc)

        response = self.model.generate_content(context_text)

        if not response.text:
            raise ValueError("Empty response from Gemini")

        result = parse_json(response.text)
        if result == "ERR_SYNTAX" or not result:
            raise ValueError("incomplete JSON format.")

        return self._normalize_answer(result)
