import json
import time
import openai
import requests
# from openai.error import RateLimitError
from random import choice
import re
# from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError
from utils import ModelEnum
import tiktoken
from utils import DataUtil
from openai import OpenAI
import aiohttp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

class GPT:
    def __init__(self, model, temperature, max_length):
        self.keys = "sk-c292bca281fe4951bfe7427c08c29f2e"  # 替换为 DeepSeek 的 API 密钥
        self.model = "deepseek-chat"
        self.temperature = temperature
        self.max_length = max_length
        self.api_base = "https://api.deepseek.com"  # 替换为 DeepSeek 的 API URL
    # @retry(wait=wait_random_exponential(multiplier=10, max=60), stop=stop_after_attempt(5), reraise=True)
    # def completion_with_backoff(self, **kwargs):
    #     return openai.ChatCompletion.create(**kwargs)

    @staticmethod
    def get_api_keys():
        with open("api_key.txt", encoding='utf-8') as file:
            api_keys = file.readlines()
            api_keys = [key.strip() for key in api_keys]  # remove the '\n' at the end of each line
        return api_keys

    def send_request(self, api_key, message_history):
        try:
            # client = OpenAI(api_key="sk-c292bca281fe4951bfe7427c08c29f2e", base_url="https://api.deepseek.com")
            # response = client.chat.completions.create(
            #     model="deepseek-chat",
            #     messages=message_history,
            #     #stream=False
            # )
            # return response.choices[0].message.content

            # client = OpenAI(api_key="sk-CSOyj1FgXQ82bhQ3tZ7EuH3hoVydfEGLWSDkdtayNNYaHWV4", base_url="https://api.zhiyunai168.com/v1")
            # response = client.chat.completions.create(
            #     #model="deepseek-chat",
            #     model='gpt-4o',
            #     messages=message_history,
            #     #stream=False
            # )
            # return response.choices[0].message.content
            client = OpenAI(api_key="sk-XZ8oig3g3zxlM7zH9OFTLaYUKRC1wTALlZCOUnVAPELTyaWb", base_url="https://api.kksj.org/v1")
            # client = OpenAI(api_key="sk-CSOyj1FgXQ82bhQ3tZ7EuH3hoVydfEGLWSDkdtayNNYaHWV4",base_url="https://api.zhiyunai168.com/v1")
            response = client.chat.completions.create(
                #model="deepseek-chat",
                #model ="deepseek-v3",
                model='gpt-4o-mini-2024-07-18',
                #model='gpt-3.5-turbo',
                messages=message_history,  # 输入的历史消息
                #stream=False
            )
            return response.choices[0].message.content

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print("The response content may not be valid JSON. Please check the API response format.")
            return None
        except Exception as e:
            print(f"Error occurred: {e}")
            return None



    def run_prompts(self, prompts):
        message_history = [{"role": "system", "content": "You are a helpful assistant."}]
        for prompt in prompts:
            message_history.append({"role": "user", "content": prompt})
            output = self.send_request(self.keys, message_history)
            message_history.append({"role": "assistant", "content": output})
        return message_history

    def run_prompts_human(self, prompts):
        message_history = [{"role": "system", "content": "You are a professional developer in Python programming. Your task is to understand the code context, raise questions if needed and generate the adapted code based on the provided answers to the questions."}]
        turn = 0
        while True:
            print(f"Turn {turn}:")
            if len(prompts) > turn:
                prompt = prompts[turn]
            else:
                last_output = message_history[-1]['content']
                try:
                    print("assistant:\n" + last_output + "\n")
                except Exception as e:
                    print(e)
                # break if LLM provide the code
                if re.search(r"```", last_output) and re.search(r"def\s\w+\(", last_output):
                    break
                prompt = input("User: ")
            message_history.append({"role": "user", "content": prompt})
            output = self.send_request(message_history)
            message_history.append({"role": "assistant", "content": output})
            turn += 1
        return message_history

    def run_prompts_mac(self, prompts):
        message_history_0 = [{"role": "system", "content": "You are a professional developer in Python programming. Your task is to understand the code context, raise questions if needed and generate the adapted code based on the provided answers to the questions."}]
        message_history_1 = [{"role": "system", "content": "You are a helpful counselor with professional experience in Python programming. Your primary task is to understand the code context, review the retrieved snippets, and generate helpful answers to the provided questions."}]
        print("Turn 0:")
        # context comprehension for both LLMs
        message_history_0.append({"role": "user", "content": prompts[0]})
        message_history_1.append({"role": "user", "content": prompts[0]})
        output = self.send_request(message_history_0)
        message_history_0.append({"role": "assistant", "content": output})
        message_history_1.append({"role": "assistant", "content": output})
        print("Turn 1:")
        # LLM1 asks the questions and LLM2 answers
        message_history_0.append({"role": "user", "content": prompts[1]})
        output = self.send_request(message_history_0)
        message_history_0.append({"role": "assistant", "content": output})
        if re.search(r"```", output) and re.search(r"def\s\w+\(", output):
            return message_history_0
        message_history_1.append({"role": "user", "content": re.sub(r'QUESTION_PLACEHOLDER', lambda m: output, prompts[2])})
        output = self.send_request(message_history_1)
        message_history_1.append({"role": "assistant", "content": output})
        print("Turn 2:")
        # LLM1 generates the code based on LLM2's answer
        message_history_0.append({"role": "user", "content": re.sub(r'ANSWER_PLACEHOLDER', lambda m: output, prompts[3])})
        output = self.send_request(message_history_0)
        message_history_0.append({"role": "assistant", "content": output})
        return message_history_0

    def run_prompts_mae(self, method_name, prompts):
        message_history_0 = [{"role": "system", "content": "You are a professional developer in Python programming. Your task is to understand the code context, raise questions if needed and generate the adapted code based on the provided answers to the questions."}]
        message_history_1 = [{"role": "system", "content": "You are a helpful evaluator with professional experience in Python programming. Your task is to review the adaptation, identify any issues, and provide feedback or instructions for further refinement."}]
        print("Turn 0:")
        # context comprehension for both LLMs
        message_history_0.append({"role": "user", "content": prompts[0]})
        message_history_1.append({"role": "user", "content": prompts[0]})
        output = self.send_request(message_history_0)
        message_history_0.append({"role": "assistant", "content": output})
        message_history_1.append({"role": "assistant", "content": output})
        print("Turn 1:")
        # LLM1 asks the questions and LLM2 answers
        message_history_0.append({"role": "user", "content": prompts[1]})
        output = self.send_request(message_history_0)
        message_history_0.append({"role": "assistant", "content": output})
        method_code = DataUtil.extract_method_from_output(output, method_name)
        if method_code == '':
            evaluation_prompt = f"Please generate the adapted method {method_name} with correct signature again."
        else:
            evaluation_prompt = re.sub(r'METHOD_PLACEHOLDER', lambda m: method_code, prompts[2])
        message_history_1.append({"role": "user", "content": evaluation_prompt})
        output = self.send_request(message_history_1)
        message_history_1.append({"role": "assistant", "content": output})
        message_history_0.append({"role": "user", "content": re.sub(r'ISSUE_PLACEHOLDER', lambda m: output, prompts[3])})
        output = self.send_request(message_history_0)
        message_history_0.append({"role": "assistant", "content": output})
        return message_history_0
