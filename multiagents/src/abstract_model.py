from abc import ABC, abstractmethod

class BaseLLMDebater(ABC):
    """
    Common interface for all debate models.
    Every model must implement gen_ans with this signature.
    """
    def __init__(self, name: str, dataset: str = "SQA"):
        self.name = name
        self.dataset = dataset

    @abstractmethod
    def gen_ans(self, sample, convincing_samples=None,
                additional_instruc=None, intervene=False) -> dict:
        """
        Return a dict like:
        {
            "reasoning": "...",
            "answer": "...",
            "confidence_level": "..."
        }
        """
        ...

    def _normalize_answer(self, result: dict) -> dict:
        """Apply dataset-specific normalization."""
        if not result:
            result = invalid_result(self.dataset)

        if self.dataset == "SQA":
            result['answer'] = str(result['answer']).lower()
        elif self.dataset == "Aqua":
            result['answer'] = str(result['answer']).upper()
        elif self.dataset in ["GSM8k", "ECQA"]:
            result['answer'] = str(result['answer'])
        return result
