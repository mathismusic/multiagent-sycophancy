# given a question, build a prompt string (basically just templates)

import os
class BasePromptBuilder:
    def __init__(self):
        pass

    def build_prompt(self, item: dict) -> str:
        """
        Build a prompt string from the given item using specific template.
        Args:
            item (dict): A dictionary containing question details.
        Returns:
            str: The constructed prompt string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class MCQPromptBuilder(BasePromptBuilder):
    def __init__(self):
        self.prompt_template = open("prompt_templates/mcq_v0.txt", "r", encoding="utf-8").read()

    def build_prompt(self, item):
        question_text = item["question"]
        choices_text = item["choices"]
        return self.prompt_template.format(question_text=question_text, choices_text=choices_text)