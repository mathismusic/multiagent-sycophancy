import os
import time
import backoff
from tqdm import tqdm
from dotenv import load_dotenv
from utils import prepare_context, prepare_context_for_chat_assistant, prepare_context_for_bard, parse_json, invalid_result

load_dotenv()
import openai
# from claude import Client
import google.generativeai as genai

from curl_cffi import CurlError
from json import JSONDecodeError
from requests.exceptions import RequestException
from google.api_core.exceptions import ServiceUnavailable
from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, InvalidRequestError


from anthropic import Anthropic
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
# openai.api_type = "azure"
# openai.api_base = os.environ['OPEN_AI_API_BASE']
# openai.api_version = os.environ['OPEN_AI_API_VERSION']
# openai.api_key = os.environ['OPEN_AI_API_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']
genai.configure(api_key=os.environ['PALM_API_KEY'])
# Note: you can add more account in .env and here
# claude_coockies = [c for c in [os.environ['CLAUDE_COOCKIE1'], os.environ['CLAUDE_COOCKIE2'], os.environ['CLAUDE_COOCKIE3'], os.environ['CLAUDE_COOCKIE4'], os.environ['CLAUDE_COOCKIE5']] if c]

class ClaudeModel:
    def __init__(self):
        # self.coockies = claude_coockies
        # self.claude_api = self.connect()
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        
    @backoff.on_exception(backoff.expo, (CurlError, RequestException), max_tries=5)   
    def connect(self):
        for coockie in self.coockies:
            claude_api = Client(coockie)
            uuid = claude_api.create_new_chat()['uuid']
            output = claude_api.send_message("Hi, what is your name and where are you from?", uuid)
            if output:
                print(output)
                print("claude connected successfully.")
                return claude_api
        print("all coockies are not available now.")
        return None
        
    @backoff.on_exception(backoff.expo, (CurlError, RequestException, ValueError), max_tries=3)
    def claude_gen_ans(self, sample, convincing_samples=None, additional_instruc=None, intervene=False, dataset="SQA"):
        contexts = prepare_context(sample, convincing_samples, intervene, dataset)

        if additional_instruc:
            contexts += " ".join(additional_instruc)

        # ---- NEW Anthropic API call ----
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": contexts}]
        )
        output = response.content[0].text
        # --------------------------------

        result = parse_json(output)
        if result == "ERR_SYNTAX" or not result:
            result = invalid_result(dataset)

        if dataset == "SQA":
            result['answer'] = result['answer'].lower()
        elif dataset == "Aqua":
            result['answer'] = result['answer'].upper()
        elif dataset in ["GSM8k", "ECQA"]:
            result['answer'] = str(result['answer'])

        return result


    def claude_debate(self, test_samples, all_results, rounds, convincing_samples, dataset):
        r = '_' + str(rounds-1)
        result = None
        for i, s in tqdm(enumerate(all_results)):
            if 'claude_output_'+str(rounds) not in s and 'debate_prompt'+ r in s and len(s['debate_prompt'+r]):
                additional_instruc = ["\n\nCarefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question."]
                additional_instruc.append("Clearly states that which pointview do you agree or disagree and why.\n\n")
                additional_instruc.append(s['debate_prompt'+r])
                additional_instruc.append("Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format.")
                try:
                    result = self.claude_gen_ans(test_samples[i],
                                                convincing_samples=convincing_samples,
                                                additional_instruc=additional_instruc,
                                                intervene=False,
                                                dataset=dataset)
                except ValueError:
                    print(result)
                    s['claude_output_'+str(rounds)] = s['claude_output'+r]
                if result != 403:
                    s['claude_output_'+str(rounds)] = result
                else:
                    print("taking a rest for the count down...")
                    break
        return all_results

@backoff.on_exception(backoff.expo, (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, ValueError), max_tries=5)
def gpt_gen_ans(sample, convincing_samples=None, additional_instruc=None, intervene=False, dataset="SQA"):
    contexts = prepare_context_for_chat_assistant(sample, convincing_samples, intervene, dataset)
    if additional_instruc:
        contexts[-1]['content'] += " ".join(additional_instruc)
    # print(contexts)
    completion = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=contexts)
    
    output = completion['choices'][0]['message']['content']
    if output:
        if "{" not in output or "}" not in output:
            raise ValueError("cannot find { or } in the model output.")
        result = parse_json(output)
        if result == "ERR_SYNTAX":
            raise ValueError("incomplete JSON format.")
            
    if not result:
        result = invalid_result(dataset)
        
    if dataset == "SQA":
        result['answer'] = result['answer'].lower()
    elif dataset == "Aqua":
        result['answer'] = result['answer'].upper()
    elif dataset in ["GSM8k", "ECQA"]:
        result['answer'] = str(result['answer'])

    return result

@backoff.on_exception(backoff.expo, (ServiceUnavailable, ValueError, TypeError), max_tries=5)
@backoff.on_exception(backoff.expo, (ServiceUnavailable, ValueError, TypeError), max_tries=5)
def bard_gen_ans(sample, convincing_samples=None, additional_instruc=None, intervene=False, dataset="SQA"):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["PALM_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")

    msg, cs, us = prepare_context_for_bard(sample, convincing_samples, intervene, dataset)

    # convert tuples (prompt, response) into readable text
    context_text = ""
    for c in cs + us:
        if isinstance(c, tuple):
            context_text += f"User: {c[0]}\nAssistant: {c[1]}\n"
        else:
            context_text += str(c) + "\n"

    if additional_instruc:
        context_text += " ".join(additional_instruc)

    response = model.generate_content(context_text)

    if not response.text:
        raise ValueError("Empty response from Gemini")

    result = parse_json(response.text)
    if result == "ERR_SYNTAX":
        raise ValueError("incomplete JSON format.") 

    if dataset == "SQA":
        result['answer'] = result['answer'].lower()
    elif dataset == "Aqua":
        result['answer'] = result['answer'].upper()
    elif dataset in ["GSM8k", "ECQA"]:
        result['answer'] = str(result['answer'])

    return result


def gpt_debate(test_samples, all_results, rounds, convincing_samples, dataset):
    r = '_' + str(rounds-1)
    for i, s in tqdm(enumerate(all_results)):
        if 'gpt3_output_'+str(rounds) not in s and 'debate_prompt'+ r in s and len(s['debate_prompt'+r]):
            additional_instruc = ["\n\nCarefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question."]
            additional_instruc.append("Clearly states that which pointview do you agree or disagree and why.\n\n")
            additional_instruc.append(s['debate_prompt'+r])
            additional_instruc.append("Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format.")
            result = gpt_gen_ans(test_samples[i],
                                 convincing_samples=convincing_samples,
                                 additional_instruc=additional_instruc,
                                 intervene=False,
                                 dataset=dataset)            
            s['gpt3_output_'+str(rounds)] = result
    return all_results

def bard_debate(test_samples, all_results, rounds, convincing_samples, dataset):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["PALM_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")

    r = '_' + str(rounds - 1)
    for i, s in tqdm(enumerate(all_results)):
        if 'bard_output_' + str(rounds) not in s and 'debate_prompt' + r in s and len(s['debate_prompt' + r]):
            additional_instruc = [
                "\n\nCarefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question.",
                "Clearly state which point of view you agree or disagree with and why.\n\n",
                s['debate_prompt' + r],
                "Output your answer in JSON format, as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format."
            ]
            try:
                result = bard_gen_ans(
                    test_samples[i],
                    convincing_samples=convincing_samples,
                    additional_instruc=additional_instruc,
                    intervene=False,
                    dataset=dataset
                )
            except ValueError:
                print("cannot generate valid answer for this sample.")
                result = invalid_result(dataset)

            s['bard_output_' + str(rounds)] = result
            time.sleep(1)
    return all_results


def bard_transform_json(model_output, dataset):
    prompt = "Transform the following paragraph to fit in the JSON format: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}"
    prompt += "Place all of the reasoning of why the answer is derived in \"reasoning\" field."
    if dataset == "SQA":
        prompt += "Place only yes or no in the \"answer\" field."
    elif dataset=="GSM8k":
        prompt += "Place only a single numeric value in the \"answer\" field."
    elif dataset=="ECQA":
        prompt += "Place only 1,2,3,4,5 representing your choice in the \"answer\" field."
    elif dataset=="Aqua":
        prompt += "Place only A,B,C,D,E representing your choice in the \"answer\" field."
    prompt += "Place the confidence level in the \"confidence_level\" field."
    prompt += model_output
    
    response = palm.chat(messages=prompt)
    return response.last

