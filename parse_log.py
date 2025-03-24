import re
import json
from collections import defaultdict


def parse_test_log(log_file):
    """解析测试日志文件"""
    results = []
    with open(log_file, 'r', encoding='utf-8') as file:
        log_content = file.read()
    # 匹配 Traceback 信息
    traceback_pattern = re.compile(
        r"Traceback \(most recent call last\):\n",
        re.MULTILINE
    )

    # 匹配 File 行
    file_line_pattern = re.compile(
        r'^\s*File "(?P<filename>[^"]+)", line (?P<line>\d+), in (?P<method>[\w.]+)(?:\n\s*File ".*?")*',
        re.MULTILINE
    )

    # 匹配错误类型和错误消息
    error_pattern = re.compile(
        r"(?P<error_type>\w+Error):\s*(?P<errormessage>.+)",
        re.DOTALL | re.MULTILINE
    )

    # 提取 Traceback 部分
    traceback_matches = traceback_pattern.finditer(log_content)
    for traceback_match in traceback_matches:
        traceback_start = traceback_match.end()
        traceback_end = log_content.find('\n\n', traceback_start)
        stacktrace = log_content[traceback_start:traceback_end]

        # 提取 File 行信息
        file_line_match = file_line_pattern.search(stacktrace)
        if file_line_match:
            filename = file_line_match.group('filename')
            line = file_line_match.group('line')
            method = file_line_match.group('method')

            # 提取错误信息
        error_match = error_pattern.search(stacktrace)
        if error_match:
            error_type = error_match.group('error_type')
            errormessage = error_match.group('errormessage').strip()
        else:
            error_type = None
            errormessage = None

        results.append({
            "File": filename,
            "Line": line,
            "Method": method,
            "Error Type": error_type,
            "Error Message": errormessage
        })

    return results


def write_failures_to_json1(failures):
    failures_list = []
    # 遍历 failures 列表并将其转换为字典格式
    for failure in failures:
        failures_list.append({
            'File': failure['File'],
            'Line': failure['Line'],
            'Method': failure['Method'],
            'Error Type': failure['Error Type'],
            'Error Message': failure['Error Message']
        })
    return failures_list
    # 将失败信息列表写入 JSON 文件
    # with open(f'failures.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(failures_list, json_file, ensure_ascii=False, indent=4)
    #
    # print(f"失败信息已成功存储到failures.json 文件中")


def parse_method(method_name):
    # 使用正则表达式匹配test和数字之间的内容
    match = re.search(r'ClassEval_(.*?)_\d+\.py', method_name)
    if match:
        # 如果匹配成功，返回filter部分，如果没有filter部分则返回空字符串
        return match.group(1)
    else:
        # 如果匹配失败，直接返回原始方法名
        return method_name


def convert_to_regular_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_to_regular_dict(v) for k, v in d.items()}
    return d


def write_failures_to_json(failures, name):
    # 使用 defaultdict 来存储失败信息
    failures_info_list = defaultdict(lambda: {'files': []})

    for failure in failures:
        method_name = parse_method(failure['File'])
        file_name = failure['File']

        # 检查是否已经存在该文件的信息
        file_info_exists = False
        for file_info in failures_info_list[method_name]['files']:
            if file_info['filename'] == file_name:
                file_info['errors'].append({
                    'Line': failure['Line'],
                    'Error Type': failure['Error Type'],
                    'Error Message': failure['Error Message']
                })
                file_info_exists = True
                break

        if not file_info_exists:
            with open(file_name, 'r', encoding='utf-8') as f:
                code_info = f.read()
            failures_info_list[method_name]['files'].append({
                'filename': file_name,
                'code': code_info,
                'errors': [
                    {
                        'Line': failure['Line'],
                        'Error Type': failure['Error Type'],
                        'Error Message': failure['Error Message']
                    }
                ]
            })

    # 将 defaultdict 转换为普通的字典
    regular_dict = convert_to_regular_dict(failures_info_list)

    # 将字典转换为列表
    result_list = [{'method': method, 'files': files} for method, info in regular_dict.items() for files in [info['files']]]

    # 写入 JSON 文件
    with open(f'{name}_failures.json', 'w', encoding='utf-8') as json_file:
        json.dump(result_list, json_file, ensure_ascii=False, indent=4)

    print(f"失败信息已成功存储到 {name}_failures.json 文件中")



# 示例调用
if __name__ == "__main__":
    log_file_path = 'log/GPT_3_5_Adaptation_test_instruction_0_task0_ctx1_woCoT_api_temp8_log_output.log'
    failures = parse_test_log(log_file_path)
    write_failures_to_json(failures,'test_instruction_0')
#     # write_failures_to_json(failures, 'instruction2')
#
#     # for failure in failures:
#     #     print(f"File: {failure['File']}")
#     #     print(f"Line: {failure['Line']}")
#     #     print(f"Method: {failure['Method']}")
#     #     # print(f"Class Name: {failure['class_name']}")
#     #     print(f"Error Type: {failure['Error Type']}")
#     #     print(f"Error Message: {failure['Error Message']}")
#     #     print("----------------------------------------")
#     write_failures_to_json(failures, 'instruction_0')



