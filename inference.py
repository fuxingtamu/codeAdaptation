import json
import time
import os
import random
from model import GPT
from prompt_loader import PromptLoader
from CodeBLEU import calc_code_bleu
from tqdm import tqdm
from utils import DataUtil, PathUtil, ModelEnum
import concurrent.futures
import threading
import copy
class Inference:
    def __init__(self, args):
        self.data = DataUtil.load_data(args.data_path, args.start_idx, args.end_idx)
        self.model = args.model
        self.temperature = args.temperature
        self.max_length = args.max_length
        self.task = args.task
        self.pattern = args.pattern
        self.task_level = args.task_level
        self.context_level = args.context_level
        self.cot_level = args.cot_level
        self.repeat = args.repeat
        self.load_from_local = args.load_from_local
        self.cuda = args.cuda
        self.lock = threading.Lock()
        self.source = args.source
        self.select = args.select
        self.name_string = PathUtil.get_name_string(self.model, self.task, self.pattern, self.task_level,
                                                    self.context_level, self.cot_level, self.source, self.temperature)
        self.loaded = self.load_model()
        if self.pattern == 'human':
            self.interactive = 1
        elif self.pattern == 'mac':
            self.interactive = 2
        elif self.pattern == 'mae':
            self.interactive = 3
        else:
            self.interactive = 0

    def load_model(self):
        if self.model == ModelEnum.GPT_3_5.value or self.model == ModelEnum.GPT_4.value:
            return GPT(self.model, self.temperature, self.max_length)

    def generate_output(self, method_name, prompts, interactive=0):
        # if len(prompt) > 2048 and self.model != ModelEnum.GPT_3_5.value:
        #     print("Prompt length exceeds the limit, truncated to 2000 characters.")
        #     prompt = prompt[:2000]
        if interactive == 0:
            output = self.loaded.run_prompts(prompts)
        elif interactive == 1:
            output = self.loaded.run_prompts_human(prompts)
        elif interactive == 2:
            output = self.loaded.run_prompts_mac(prompts)
        elif interactive == 3:
            output = self.loaded.run_prompts_mae(method_name, prompts)
        else:
            raise ValueError("Interactive mode not supported.")
        return output

    @staticmethod
    def select_code_snippet(code_list, mode):
        if mode == 'random':
            random.seed(0)
            return random.randint(0, len(code_list) - 1)
        elif mode == 'test':
            return 0
        elif mode == 'codebleu':
            return 0
        elif mode == 'codebertscore':
            return 0
        pass

    def process_single_call(self, method_name, prompts, r):
        if self.model == ModelEnum.GPT_3_5.value:
            time.sleep(3)  # Rate limit: 3 requests per minute
        elif self.model == ModelEnum.GPT_4.value:
            time.sleep(30)  # TO-DO: Check the rate limit
        message_history = self.generate_output(method_name, prompts, self.interactive)
        if self.model != ModelEnum.GPT_3_5.value:
            output = message_history
            if r == 0:
                try:
                    print(f'Prompt: {prompts[0]}\n')
                except Exception as e:
                    print(e)
        else:
            output = message_history[-1]["content"]
            for m in message_history:
                try:
                    print(m['role'] + ":\n" + m['content'] + "\n")
                except Exception as e:
                    print(e)
        return output

    def process_method(self, method, case, method_name, method_idx):
        tmp_prompts = []
        tmp_raw_output = []
        tmp_predicted = []
        tmp_codebleu = []
        # generate one or adapt a selected code snippet
        predicted_idx = self.select_code_snippet(method['predicted'], self.select) if 'predicted' in method else 0

        prompt_loader = PromptLoader(case, method_idx, predicted_idx, self.model, self.task, self.pattern, self.repeat)
        prompts = prompt_loader.generate_prompt(self.task_level, self.context_level, self.cot_level, case)
        # if self.model == ModelEnum.CodeGeex.value:
        #     prompt = DataUtil.convert_prompt_to_comment(prompt)
        if prompts is None:
            #说明前一轮的结果没问题，可以直接不用再询问大模型，直接返回上次的结果
            #上一轮的结果根据self.pattern进行判断
            if self.pattern == 'test_instruction_0':
                #现在要去之前的结果中提取最关键的predicted_code
                with open('output/GPT_3_5_Adaptation_instruction_task0_ctx1_woCoT_api_temp8_output.json', 'r', encoding='utf-8') as f:
                    prev_output = json.load(f)
                    prev_output_dict = {d['task_id']: d for d in prev_output}
                    prev_method = prev_output_dict[case['task_id']]['methods_info'][method_idx]

                tmp_predicted = prev_method['predicted']

                tmp_prompts = 'The previous one is correct, there is no need to revise it.'
                tmp_raw_output=str(prev_method['raw_output'])
                method['prompt'] = tmp_prompts
                method['raw_output'] = tmp_raw_output

                method_code = DataUtil.extract_method_from_output(tmp_raw_output, method_name)
                tmp_codebleu.append(
                    calc_code_bleu.codebleu([DataUtil.remove_all_comments(method['solution_code'])],
                                            [DataUtil.remove_all_comments(method_code)], 'python'))

                method['predicted'] = tmp_predicted
                method['codebleu'] = tmp_codebleu
                method['idx'] = method_idx
                # self.save_tmp_results(case, case['task_id'], method_name)

            elif self.pattern == 'test_instruction_1':
                with open('output/GPT_3_5_Adaptation_test_instruction_0_task0_ctx1_woCoT_api_temp8_output.json', 'r', encoding='utf-8') as f:
                    prev_output = json.load(f)
                    prev_output_dict = {d['task_id']: d for d in prev_output}
                    prev_method = prev_output_dict[case['task_id']]['methods_info'][method_idx]

                tmp_predicted = prev_method['predicted']
                tmp_prompts = 'The previous one is correct, there is no need to revise it.'

                tmp_raw_output=str(prev_method['raw_output'])
                method['prompt'] = tmp_prompts
                method['raw_output'] = tmp_raw_output

                method_code = DataUtil.extract_method_from_output(tmp_raw_output, method_name)
                tmp_codebleu.append(
                    calc_code_bleu.codebleu([DataUtil.remove_all_comments(method['solution_code'])],
                                            [DataUtil.remove_all_comments(method_code)], 'python'))
                method['predicted'] = tmp_predicted
                method['codebleu'] = tmp_codebleu
                method['idx'] = method_idx
                # self.save_tmp_results(case, case['task_id'], method_name)

            elif self.pattern == 'test_instruction_2':
                with open('output/GPT_3_5_Adaptation_test_instruction_1_task0_ctx1_woCoT_api_temp8_output.json', 'r', encoding='utf-8') as f:
                    prev_output = json.load(f)
                    prev_output_dict = {d['task_id']: d for d in prev_output}
                    prev_method = prev_output_dict[case['task_id']]['methods_info'][method_idx]

                tmp_predicted = prev_method['predicted']
                tmp_prompts = 'The previous one is correct, there is no need to revise it.'
                tmp_raw_output=str(prev_method['raw_output'])
                method['prompt'] = tmp_prompts
                method['raw_output'] = tmp_raw_output

                method_code = DataUtil.extract_method_from_output(tmp_raw_output, method_name)
                tmp_codebleu.append(
                    calc_code_bleu.codebleu([DataUtil.remove_all_comments(method['solution_code'])],
                                            [DataUtil.remove_all_comments(method_code)], 'python'))
                method['predicted'] = tmp_predicted
                method['codebleu'] = tmp_codebleu
                method['idx'] = method_idx
                # self.save_tmp_results(case, case['task_id'], method_name)

            elif self.pattern == 'test_instruction_3':
                with open('output/GPT_3_5_Adaptation_test_instruction_2_task0_ctx1_woCoT_api_temp8_output.json', 'r', encoding='utf-8') as f:
                    prev_output = json.load(f)
                    prev_output_dict = {d['task_id']: d for d in prev_output}
                    prev_method = prev_output_dict[case['task_id']]['methods_info'][method_idx]

                tmp_predicted = prev_method['predicted']
                tmp_prompts = 'The previous one is correct, there is no need to revise it.'
                tmp_raw_output=str(prev_method['raw_output'])
                method['prompt'] = tmp_prompts
                method['raw_output'] = tmp_raw_output

                method_code = DataUtil.extract_method_from_output(tmp_raw_output, method_name)
                tmp_codebleu.append(
                    calc_code_bleu.codebleu([DataUtil.remove_all_comments(method['solution_code'])],
                                            [DataUtil.remove_all_comments(method_code)], 'python'))
                method['predicted'] = tmp_predicted
                method['codebleu'] = tmp_codebleu
                method['idx'] = method_idx

            elif self.pattern == 'test_instruction_4':
                with open('output/GPT_3_5_Adaptation_test_instruction_3_task0_ctx1_woCoT_api_temp8_output.json', 'r', encoding='utf-8') as f:
                    prev_output = json.load(f)
                    prev_output_dict = {d['task_id']: d for d in prev_output}
                    prev_method = prev_output_dict[case['task_id']]['methods_info'][method_idx]

                tmp_predicted = prev_method['predicted']
                tmp_prompts = 'The previous one is correct, there is no need to revise it.'
                tmp_raw_output=str(prev_method['raw_output'])
                method['prompt'] = tmp_prompts
                method['raw_output'] = tmp_raw_output

                method_code = DataUtil.extract_method_from_output(tmp_raw_output, method_name)
                tmp_codebleu.append(
                    calc_code_bleu.codebleu([DataUtil.remove_all_comments(method['solution_code'])],
                                            [DataUtil.remove_all_comments(method_code)], 'python'))
                method['predicted'] = tmp_predicted
                method['codebleu'] = tmp_codebleu
                method['idx'] = method_idx

        else:
            print(f'\nAdapting the snippet at index {predicted_idx}.\n')

            with concurrent.futures.ThreadPoolExecutor() as executor:  # 并行化
                futures = [executor.submit(self.process_single_call, method_name, prompts, r) for r in
                            range(self.repeat)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        output = future.result()
                        method_code = DataUtil.extract_method_from_output(output, method_name)
                        tmp_prompts.append('\n'.join(prompts))
                        tmp_raw_output.append(output)
                        tmp_predicted.append(method_code)
                        tmp_codebleu.append(
                            calc_code_bleu.codebleu([DataUtil.remove_all_comments(method['solution_code'])],
                                                        [DataUtil.remove_all_comments(method_code)], 'python'))
                    except Exception as e:
                        print(f"生成请求出错: {e}")
                method['prompt'] = tmp_prompts
                method['raw_output'] = tmp_raw_output
                method['predicted'] = tmp_predicted
                method['codebleu'] = tmp_codebleu
                method['idx'] = method_idx
                    # method_idx += 1
            self.save_tmp_results(case, case['task_id'], method_name)
        return method

    def pipeline(self):
        results = []
        for case in tqdm(self.data):
            method_idx = 0
            method_info = case['methods_info']
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_method, method, case, method['method_name'], idx) for idx, method in enumerate(method_info)]
                for future in concurrent.futures.as_completed(futures):
                    # try:
                    method = future.result()
                    with self.lock:
                        case['methods_info'][method['idx']]=method
                    # except KeyError:
                    #     print("Method does not contain 'idx' key:", method)
                    # except Exception as e:
                    #     print(f"处理方法出错: {e}")
            # for method in case['methods_info']:
            #     method_name = method['method_name']
            #     tmp_prompts = []
            #     tmp_raw_output = []
            #     tmp_predicted = []
            #     tmp_codebleu = []
            #     # generate one or adapt a selected code snippet
            #     predicted_idx = self.select_code_snippet(method['predicted'], self.select) if 'predicted' in method else 0
            #
            #     prompt_loader = PromptLoader(case, method_idx, predicted_idx, self.model, self.task, self.pattern)
            #     prompts = prompt_loader.generate_prompt(self.task_level, self.context_level, self.cot_level)
            #     # if self.model == ModelEnum.CodeGeex.value:
            #     #     prompt = DataUtil.convert_prompt_to_comment(prompt)
            #
            #     print(f'\nAdapting the snippet at index {predicted_idx}.\n')
            #
            #     with concurrent.futures.ThreadPoolExecutor() as executor:#并行化
            #         futures = [executor.submit(self.process_single_call, method_name, prompts, r) for r in
            #                    range(self.repeat)]
            #         for future in concurrent.futures.as_completed(futures):
            #             try:
            #                 output = future.result()
            #                 method_code = DataUtil.extract_method_from_output(output, method_name)
            #                 tmp_prompts.append('\n'.join(prompts))
            #                 tmp_raw_output.append(output)
            #                 tmp_predicted.append(method_code)
            #                 tmp_codebleu.append(
            #                     calc_code_bleu.codebleu([DataUtil.remove_all_comments(method['solution_code'])],
            #                                             [DataUtil.remove_all_comments(method_code)], 'python'))
            #             except Exception as e:
            #                 print(f"生成请求出错: {e}")

                # for r in range(self.repeat):
                #     if self.model == ModelEnum.GPT_3_5.value:
                #         time.sleep(3)  # rate limit: 3 requests per minute
                #     elif self.model == ModelEnum.GPT_4.value:
                #         time.sleep(30)  # TO-DO: check the rate limit
                #     message_history = self.generate_output(method_name, prompts, self.interactive)
                #     if self.model != ModelEnum.GPT_3_5.value:
                #         output = message_history
                #         if r == 0:
                #             try:
                #                 print(f'Prompt: {prompts[0]}\n')
                #             except Exception as e:
                #                 print(e)
                #     else:
                #         output = message_history[-1]["content"]
                #         #if r == 0:
                #         for m in message_history:
                #             try:
                #                 print(m['role'] + ":\n" + m['content'] + "\n")
                #             except Exception as e:
                #                 print(e)
                #     method_code = DataUtil.extract_method_from_output(output, method_name)
                #     tmp_prompts.append('\n'.join(prompts))
                #     tmp_raw_output.append(output)
                #     tmp_predicted.append(method_code)
                #     tmp_codebleu.append(calc_code_bleu.codebleu([DataUtil.remove_all_comments(method['solution_code'])],
                #                                                 [DataUtil.remove_all_comments(method_code)], 'python'))
                #     method['prompt'] = tmp_prompts
                #     method['raw_output'] = tmp_raw_output
                #     method['predicted'] = tmp_predicted
                #     method['codebleu'] = tmp_codebleu
                # method_idx += 1
                # self.save_tmp_results(case, case['task_id'], method_name)
            results.append(case)
        self.save_results(results)
        # self.tear_down()

    def save_results(self, res):
        output_path = PathUtil.get_output_path(self.name_string)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4)

    def save_tmp_results(self, tmp_res, tmp_idx, method_name):
        tmp_res_copy = copy.deepcopy(tmp_res)
        tmp_output_path = PathUtil.get_tmp_output_path(self.name_string, tmp_idx, method_name)
        with open(tmp_output_path, 'w', encoding='utf-8') as f:
            json.dump(tmp_res_copy, f, indent=4)

    def tear_down(self):
        file_list = os.listdir("./tmp_output")
        tmp_output_prefix = f"{self.name_string}_tmp"
        for item in file_list:
            if item.startswith(tmp_output_prefix):
                os.remove(os.path.join("./tmp_output", item))
