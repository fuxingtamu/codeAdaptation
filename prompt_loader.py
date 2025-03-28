import json
import random

from utils import PromptUtils, DataUtil, TaskEnum, ModelEnum


class PromptLoader:
    def __init__(self, data, method_idx, predicted_idx, model, task, pattern, repeat):
        self.data = data
        self.method_idx = method_idx
        self.predicted_idx = predicted_idx
        self.model = model
        self.task = task
        self.pattern = pattern
        self.repeat = repeat

    def generate_task(self, task_level):
        method = self.data['methods_info'][self.method_idx]
        if self.task == TaskEnum.Method_Generation.value:
            method_name = method['method_name']
        else:
            predicted_method = self.data['methods_info'][self.method_idx]['predicted'][self.predicted_idx]
            predicted_name = DataUtil.extract_method_name(predicted_method)
            if predicted_name != '':
                method_name = predicted_name
            else:
                print("Predicted method name is empty, use the original method name instead.")
                method_name = method['method_name']
        method_description = method['method_description']
        if self.model == ModelEnum.CodeGeex.value:
            if self.task == TaskEnum.Method_Generation.value:
                return PromptUtils.method_generation_task_cc(method_name, method_description, self.data['class_name'], task_level)
            else:
                raise Exception(f"Unknown task type.")
        else:
            if self.task == TaskEnum.Method_Generation.value:
                return PromptUtils.method_generation_task(method_name, method_description, self.data['class_name'], task_level)
            elif self.task == TaskEnum.Adaptation.value:
                return PromptUtils.adaptation_task(method_name, method_description, self.data['class_name'], task_level)
            else:
                raise Exception(f"Unknown task type.")


    def generate_context(self, context_level, inplace=False, with_description=False):
        context = ""
        if context_level > 0:
            # generate context comment
            context += "### Class Context\n"
            if context_level == 1:
                context += 'For member methods in the following class context, only their signatures are retained.\n'
            elif context_level == 2:
                context += 'For member methods in the following class context, only their descriptions are retained.\n'
            else:
                pass

            # class-level context
            class_context = PromptUtils.class_context_classeval(self.data['import_statement'],
                                                                self.data['class_constructor'],
                                                                self.data['class_description'])
            # method-level context
            method_context = ""
            target_method_name = self.data['methods_info'][self.method_idx]['method_name']
            if self.task == TaskEnum.Method_Generation.value:
                target_method_code = self.generate_target_method()
            else:
                # for fix scenarios, use: target_method_code = self.generate_provided_method()
                target_method_code = ""
            for method in self.data['methods_info']:
                method_name = method['method_name']
                # skip the target method at its position if inplace is False
                if method_name == target_method_name:
                    if inplace is True:
                        method_context += target_method_code + "\n\n"
                    else:
                        continue
                method_description = method['method_description']
                method_code = method['solution_code']
                if context_level == 1:  # only method signature
                    method_context += "    " + DataUtil.extract_method_signature(method_name, method_description) + \
                                      "\n        pass\n\n"
                elif context_level == 2:  # method signature + example input/output
                    method_context += "    " + method_description + "\n        pass\n\n"
                elif context_level == 3:  # method code (only independent ones)
                    target_method = self.data['methods_info'][self.method_idx]
                    if DataUtil.is_dependent(method, target_method) == 1:
                        continue
                    else:
                        method_context += "    " + method_code + "\n\n"
                else:  # all method code
                    method_context += "    " + method_code + "\n\n"
            # traditional generation or code completion scenario
            if self.model == ModelEnum.CodeGeex.value:
                context = '\n'.join(['# ' + line for line in context.split('\n')[1:]])
                context += class_context + "\n" + method_context
            else:
                # concatenate the target method
                if inplace is False:
                    context += "```\n" + class_context + "\n" + method_context + target_method_code + "\n```\n"
                else:
                    context += "```\n" + class_context + "\n" + method_context + "```\n"
            if with_description is True:
                class_name = self.data['class_name']
                context += f"Above is a class context of `{class_name}`, please understand and remember it but do " \
                           f"not reply to this message, do you understand?\n"
        return context

    def generate_provided_method(self, with_caption=False, with_description=False):
        """
        Generate the provided method for adaptation or bug fix task.
        """
        if self.task == TaskEnum.Adaptation.value:
            provided_method = self.data['methods_info'][self.method_idx]['predicted'][self.predicted_idx]
            target_method = self.data['methods_info'][self.method_idx]
            target_method_name = target_method['method_name']
            method_description = target_method['method_description']
            method_desc_dict = DataUtil.parse_method_description(target_method_name, method_description)
            method_function = method_desc_dict['function']
            if with_description:
                description = f"Above is a code snippet that I found to implement the function: {method_function}. " \
                              f"Note that it is not necessarily correct. Please understand and remember it but do not " \
                              f"reply to this message, do you understand?"
            else:
                description = ""
            if with_caption:
                return f"### Provided Method\n```\n{provided_method}\n```\n{description}\n"
            else:
                return provided_method
        elif self.task == TaskEnum.Method_Generation.value:
            return ""
        else:
            raise Exception(f"Unknown task type.")

    def generate_target_method(self):
        """
        Generate the target method to be completed.
        """
        return "    " + self.data['methods_info'][self.method_idx]['method_description'] + "\n        pass"

    def generate_steps(self, cot_level):
        restriction_dependencies = self.generate_dependencies()
        if self.task == TaskEnum.Method_Generation.value:
            return PromptUtils.generation_step(cot_level, restriction_dependencies)
        elif self.task == TaskEnum.Adaptation.value:
            return PromptUtils.adaptation_step(cot_level, restriction_dependencies)
        else:
            raise Exception(f"Unknown task type.")

    def generate_dependencies(self):
        """
        Generate the dependencies for the target method.
        """
        return PromptUtils.restriction_dependencies(self.data['methods_info'][self.method_idx]['dependencies'])

    def get_failure_info_and_code_info(self, method_name, case_id):
        random_integer = random.randint(0, 4)
        code_info = None
        failure_info = None
        combine_method_name = case_id.split('_')[-1] + '_' + method_name
        if self.pattern == 'test_instruction_0':
            with open(f'instruction_failures.json', 'r', encoding='utf-8') as f:
                failure_init_info = json.load(f)
                for failure in failure_init_info:
                    if combine_method_name == failure['method']:
                        random_file = random.choice(failure['files'])
                        #怎么提供信息最好，随机提供还是全部提供
                        code_info = random_file['code']
                        failure_info = random_file['errors']
                        break

        elif self.pattern == 'test_instruction_1':
            with open(f'test_instruction_0_failures.json', 'r', encoding='utf-8') as f:
                failure_first_info = json.load(f)
                for failure in failure_first_info:
                    if combine_method_name == failure['method']:
                        code_info = 'second test code: \n'
                        failure_info = {}
                        first_random_file = random.choice(failure['files'])
                        # 怎么提供信息最好，随机提供还是全部提供
                        code_info += first_random_file['code']
                        failure_info['second_error'] = first_random_file['errors']
                        with open(f'instruction_failures.json', 'r', encoding='utf-8') as f1:
                            failure_init_info = json.load(f1)
                            for failure in failure_init_info:
                                if combine_method_name == failure['method']:
                                    random_file = random.choice(failure['files'])
                                    code_info += '\n\nfirst test code: \n'
                                    code_info += random_file['code']
                                    failure_info['first_error'] = random_file['errors']
                                    break
                        break

        elif self.pattern == 'test_instruction_2':
            with open(f'test_instruction_1_failures.json', 'r', encoding='utf-8') as f:
                failure_second_info = json.load(f)
                for failure in failure_second_info:
                    if combine_method_name == failure['method']:
                        code_info = 'third test code: \n'
                        failure_info = {}
                        first_random_file = random.choice(failure['files'])
                        # 怎么提供信息最好，随机提供还是全部提供
                        code_info += first_random_file['code']
                        failure_info['third_error'] = first_random_file['errors']

                        with open(f'test_instruction_0_failures.json', 'r', encoding='utf-8') as f1:
                            failure_first_info = json.load(f1)
                            for failure in failure_first_info:
                                if combine_method_name == failure['method']:
                                    random_file = random.choice(failure['files'])
                                    code_info += '\n\nsecond test code: \n'
                                    code_info += random_file['code']
                                    failure_info['second_error'] = random_file['errors']

                                    with open(f'instruction_failures.json', 'r', encoding='utf-8') as f2:
                                        failure_init_info = json.load(f2)
                                        for failure in failure_init_info:
                                            if combine_method_name == failure['method']:
                                                random_file = random.choice(failure['files'])
                                                code_info += '\n\nfirst test code: \n'
                                                code_info += random_file['code']
                                                failure_info['first_error'] = random_file['errors']
                                                break
                                    break
                        break

        elif self.pattern == 'test_instruction_3':
            with open(f'test_instruction_2_failures.json', 'r', encoding='utf-8') as f:
                failure_third_info = json.load(f)
                for failure in failure_third_info:
                    if combine_method_name == failure['method']:
                        code_info = 'forth test code: \n'
                        failure_info = {}
                        first_random_file = random.choice(failure['files'])
                        # 怎么提供信息最好，随机提供还是全部提供
                        code_info += first_random_file['code']
                        failure_info['forth_error'] = first_random_file['errors']

                        with open(f'test_instruction_1_failures.json', 'r', encoding='utf-8') as f1:
                            failure_second_info = json.load(f1)
                            for failure in failure_second_info:
                                if combine_method_name == failure['method']:
                                    random_file = random.choice(failure['files'])
                                    code_info += '\n\nthird test code: \n'
                                    code_info += random_file['code']
                                    failure_info['third_error'] = random_file['errors']

                                    with open(f'test_instruction_0_failures.json', 'r', encoding='utf-8') as f2:
                                        failure_first_info = json.load(f2)
                                        for failure in failure_first_info:
                                            if combine_method_name == failure['method']:
                                                random_file = random.choice(failure['files'])
                                                code_info += '\n\nsecond test code: \n'
                                                code_info += random_file['code']
                                                failure_info['second_error'] = random_file['errors']

                                                with open(f'instruction_failures.json', 'r',encoding='utf-8') as f3:
                                                    failure_init_info = json.load(f3)
                                                    for failure in failure_init_info:
                                                        if combine_method_name == failure['method']:
                                                            random_file = random.choice(failure['files'])
                                                            code_info += '\n\nfirst test code: \n'
                                                            code_info += random_file['code']
                                                            failure_info['first_error'] = random_file['errors']
                                                            break
                                                break
                                    break
                        break

        elif self.pattern == 'test_instruction_4':
            with open(f'test_instruction_3_failures.json', 'r', encoding='utf-8') as f:
                failure_forth_info = json.load(f)
                for failure in failure_forth_info:
                    if combine_method_name == failure['method']:
                        code_info = 'fifth test code: \n'
                        failure_info = {}
                        first_random_file = random.choice(failure['files'])
                        # 怎么提供信息最好，随机提供还是全部提供
                        code_info += first_random_file['code']
                        failure_info['fifth_error'] = first_random_file['errors']

                        with open(f'test_instruction_1_failures.json', 'r', encoding='utf-8') as f1:
                            failure_third_info = json.load(f1)
                            for failure in failure_third_info:
                                if combine_method_name == failure['method']:
                                    random_file = random.choice(failure['files'])
                                    code_info += '\n\nforth test code: \n'
                                    code_info += random_file['code']
                                    failure_info['forth_error'] = random_file['errors']

                                    with open(f'test_instruction_0_failures.json', 'r', encoding='utf-8') as f2:
                                        failure_second_info = json.load(f2)
                                        for failure in failure_second_info:
                                            if combine_method_name == failure['method']:
                                                random_file = random.choice(failure['files'])
                                                code_info += '\n\nthird test code: \n'
                                                code_info += random_file['code']
                                                failure_info['third_error'] = random_file['errors']

                                                with open(f'instruction_failures.json', 'r',encoding='utf-8') as f3:
                                                    failure_first_info = json.load(f3)
                                                    for failure in failure_first_info:
                                                        if combine_method_name == failure['method']:
                                                            random_file = random.choice(failure['files'])
                                                            code_info += '\n\nsecond test code: \n'
                                                            code_info += random_file['code']
                                                            failure_info['second_error'] = random_file['errors']

                                                            with open(f'instruction_failures.json', 'r',encoding='utf-8') as f4:
                                                                failure_init_info = json.load(f4)
                                                                for failure in failure_init_info:
                                                                    if combine_method_name == failure['method']:
                                                                        random_file = random.choice(failure['files'])
                                                                        code_info += '\n\nfirst test code: \n'
                                                                        code_info += random_file['code']
                                                                        failure_info['first_error'] = random_file['errors']
                                                                        break
                                                            break
                                                break
                                    break
                        break
        return failure_info, code_info

    def generate_prompt(self, task_level, context_level, cot_level, case):
        task_description = self.generate_task(task_level)
        steps = self.generate_steps(cot_level)
        examples = ""
        target_method = self.generate_target_method()
        provided_method_with_caption = self.generate_provided_method(with_caption=True)
        class_name = self.data['class_name']
        method_name = self.data['methods_info'][self.method_idx]['method_name']
        method_description = self.data['methods_info'][self.method_idx]['method_description']
        requirements = DataUtil.parse_method_description(method_name, method_description)
        context_information = self.generate_context(context_level)
        method_dependencies = self.data['methods_info'][self.method_idx]['dependencies']
        case_id = case['task_id']

        if self.pattern == 'instruction':
            return [PromptUtils.pattern_instruction_style(task_description + '\n' + provided_method_with_caption +
                                                         context_information + steps)]

        elif 'test_instruction' in self.pattern :
            failure_info, code_info = self.get_failure_info_and_code_info(method_name, case_id)
            if failure_info is None and code_info is None:
                return None
            else:
                suggestions = ""
                if 'AssertionError' in str(failure_info):
                    #suggestions += "The code can be compiled, but the function is not perfect. 'first error' corresponds to 'first test code', and so on. Focus on each error and the code you changed to make the test match the Assertions as much as possible."
                    suggestions += "The unit test encounters an AssertionError, which is in Code info, and the corresponding error is in Failure info. The first error corresponds to the first test code. I need you to analyze the cause of the error first, and then provide a solution. After the repair, it needs to be verified and explained."

                if 'AttributeError' in str(failure_info) or 'NameError' in str(failure_info):
                    suggestions += str(method_dependencies)
                    #suggestions += "Waring: Always import external libraries or call methods in your class as specified in method_dependencies. Never import arbitrary external libraries or use undefined methods."
                    suggestions += '\n'
                    suggestions += "If Suggestion's lib_dependencies is empty, you won't be able to import external librarys, such as numpy, pandas, matplotlib, math, pds, deque, random, radians, etc.Instead, try calling methods in Suggestion's field_dependencies and Suggestion's method_dependencies."

                if 'IndexError' in str(failure_info):
                    suggestions += "Please check the index of the input parameters of the method."
                if 'KeyError' in str(failure_info):
                    suggestions += "Please check the key of the input parameters of the method."
                if 'TypeError' in str(failure_info):
                    suggestions += "Please check the type of the input parameters of the method."
                if 'ValueError' in str(failure_info):
                    suggestions += "Please check the input parameters of the method."

                return [PromptUtils.pattern_test_instruction_style(task_description + '\n' + provided_method_with_caption +
                                                         context_information + steps, failure_info, code_info, suggestions)]

        elif self.pattern == 'direct':
            if self.model == ModelEnum.CodeGeex.value:
                return [PromptUtils.pattern_direct_style('# language: Python\n\n' + context_information
                                                        + task_description)]
            else:
                return [PromptUtils.pattern_direct_style(task_description + steps + context_information)]
        elif self.pattern == 'enhanced':
            # provide class context
            first_prompt = self.generate_context(context_level, with_description=True)
            # provide code snippet to be adapted
            provided_method_with_caption = self.generate_provided_method(with_caption=True, with_description=False)
            second_prompt = f"{provided_method_with_caption}"
            second_prompt += "Assume that you are a developer who found the above code snippet online to implement a " \
                             "feature. However, the code snippet is not necessarily correct. "
            # provide task description
            second_prompt += task_description + steps
            return [first_prompt, second_prompt]
        elif self.pattern == 'human':
            # provide class context
            first_prompt = self.generate_context(context_level, with_description=True)
            # provide code snippet to be adapted
            provided_method_with_caption = self.generate_provided_method(with_caption=True, with_description=False)
            second_prompt = provided_method_with_caption
            second_prompt += "Assume that you are a developer who found the above code snippet online to implement a " \
                             "feature. However, the code snippet is not necessarily correct. "
            # provide task description
            second_prompt += task_description + steps
            return [first_prompt, second_prompt]
        elif self.pattern == 'mac':
            first_prompt = self.generate_context(context_level, with_description=True)
            second_turn_intro = (f"I will provide you with a code snippet of the method {method_name} to be adapted "
                                 f"to the above class.\n")
            provided_method_with_caption = self.generate_provided_method(with_caption=True, with_description=False)
            second_prompt = (second_turn_intro + provided_method_with_caption + task_description + "### Questions:\n")
            third_prompt = (f"If you are going to implement a method {method_name} in the above class to fulfill "
                            f"the following requirement:\n```\n{method_description}\n```\n"
                            f"Please answer the following questions:\nQUESTION_PLACEHOLDER\n")
            fourth_prompt = (f"The answers to your questions are listed below:\nANSWER_PLACEHOLDER\n\n{steps}\n"
                             f"### Adapted Method:\n")
            return [first_prompt, second_prompt, third_prompt, fourth_prompt]
        elif self.pattern == 'mae':
            # provide class context
            first_prompt = self.generate_context(context_level, with_description=True)
            # provide code snippet to be adapted
            provided_method_with_caption = self.generate_provided_method(with_caption=True, with_description=False)
            second_prompt = f"{provided_method_with_caption}"
            second_prompt += "Assume that you are a developer who found the above code snippet online to implement a " \
                             "feature. However, the code snippet is not necessarily correct. "
            # provide task description
            second_prompt += task_description + steps + "\n\n### Adapted Method:\n"
            # enable self evaluation
            third_prompt = ("### Provided Method:\n```\nMETHOD_PLACEHOLDER\n```\n\n"
                            f"Above is a code snippet of the method {method_name} to be adapted to "
                            f"the class {class_name}. Please list any issues in the code snippet that do not conform "
                            f"to the class context and the requirement each with a sentence as a bullet in the "
                            f"Issues section, but do not make any modifications.\n"
                            "### Issues:\n")
            fourth_prompt = ("I will provide you with a list of issues identified by the evaluator in the following:\n"
                             "ISSUE_PLACEHOLDER\n\n"
                             "Based on the identified issues, please adapt the method again to the class context and "
                             "the requirement.")
            return [first_prompt, second_prompt, third_prompt, fourth_prompt]




