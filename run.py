import argparse
from inference import Inference
from test import Test
import parse_log
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/ClassEval_data.json",
        help="data path"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="start index of data"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=100,
        help="end index of data"
    )
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        help="model name, default is 0 ('gpt-3.5-turbo')"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature value in generation config"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="max length of model's generation result"
    )
    parser.add_argument(
        "--task",
        type=int,
        default=0,
        help="task type, default is 0 (method generation)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="direct",
        help="prompt pattern"
    )
    parser.add_argument(
        "--task_level",
        type=int,
        default=0,
        help="task level, default is 0 (simple task description)"
    )
    parser.add_argument(
        "--context_level",
        type=int,
        default=0,
        help="context level, default is 0 (no context)"
    )
    parser.add_argument(
        "--cot_level",
        type=int,
        default=0,
        help="CoT level, default is 0 (no steps)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat times, or number of samples to generate"
    )
    parser.add_argument(
        "--select",
        type=str,
        default='random',
        help="how to select target code snippet from multiple methods"
    )
    parser.add_argument(
        "--load_from_local",
        type=bool,
        default=True,
        help="load LLMs from local directory, default is True"
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="cuda device id, default is 0"
    )
    parser.add_argument(
        "--source",
        type=str,
        default='',
        help="adaptation source, default is None, could specify model param size"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()

    inference = Inference(args)
    inference.pipeline()

    test = Test(args)
    test_results = test.pipeline()
    test.save_results(test_results)
    pass_at_1, overview, errors = test.evaluate_test_results(test_results, 1)
    pass_at_5, _, _ = test.evaluate_test_results(test_results, 5)
    print("pass@1:", pass_at_1)
    print("pass@5:", pass_at_5)
    print("overview:", overview)
    print("errors:", errors)

    fail_init_info = parse_log.parse_test_log('log/GPT_3_5_Adaptation_instruction_task0_ctx1_woCoT_api_temp8_log_output.log')
    parse_log.write_failures_to_json(fail_init_info, 'instruction')

    #先尝试一次测试，看看能否跑通，然后再尝试多次测试
    args.pattern = 'test_instruction_0'
    inference = Inference(args)
    inference.pipeline()
    test1 = Test(args)
    test1_results = test1.pipeline()
    test1.save_results(test1_results)
    pass_at_1, overview, errors = test1.evaluate_test_results(test1_results, 1)
    pass_at_5, _, _ = test1.evaluate_test_results(test1_results, 5)
    print(args.pattern+"pass@1:", pass_at_1)
    print(args.pattern+"pass@5:", pass_at_5)
    print(args.pattern+"overview:", overview)
    print(args.pattern+"errors:", errors)
    fail_init_info = parse_log.parse_test_log(
        f'log/GPT_3_5_Adaptation_{args.pattern}_task0_ctx1_woCoT_api_temp8_log_output.log')
    parse_log.write_failures_to_json(fail_init_info, args.pattern)

    args.pattern = 'test_instruction_1'
    inference = Inference(args)
    inference.pipeline()
    test = Test(args)
    test_results = test.pipeline()
    test.save_results(test_results)
    pass_at_1, overview, errors = test.evaluate_test_results(test_results, 1)
    pass_at_5, _, _ = test.evaluate_test_results(test_results, 5)
    print(args.pattern + "pass@1:", pass_at_1)
    print(args.pattern + "pass@5:", pass_at_5)
    print(args.pattern + "overview:", overview)
    print(args.pattern + "errors:", errors)
    fail_init_info = parse_log.parse_test_log(
        f'log/GPT_3_5_Adaptation_{args.pattern}_task0_ctx1_woCoT_api_temp8_log_output.log')
    parse_log.write_failures_to_json(fail_init_info, args.pattern)

    args.pattern = 'test_instruction_2'
    inference = Inference(args)
    inference.pipeline()
    test = Test(args)
    test_results = test.pipeline()
    test.save_results(test_results)
    pass_at_1, overview, errors = test.evaluate_test_results(test_results, 1)
    pass_at_5, _, _ = test.evaluate_test_results(test_results, 5)
    print(args.pattern + "pass@1:", pass_at_1)
    print(args.pattern + "pass@5:", pass_at_5)
    print(args.pattern + "overview:", overview)
    print(args.pattern + "errors:", errors)
    fail_init_info = parse_log.parse_test_log(
        f'log/GPT_3_5_Adaptation_{args.pattern}_task0_ctx1_woCoT_api_temp8_log_output.log')
    parse_log.write_failures_to_json(fail_init_info, args.pattern)

    args.pattern = 'test_instruction_3'
    inference = Inference(args)
    inference.pipeline()
    test = Test(args)
    test_results = test.pipeline()
    test.save_results(test_results)
    pass_at_1, overview, errors = test.evaluate_test_results(test_results, 1)
    pass_at_5, _, _ = test.evaluate_test_results(test_results, 5)
    print(args.pattern + "pass@1:", pass_at_1)
    print(args.pattern + "pass@5:", pass_at_5)
    print(args.pattern + "overview:", overview)
    print(args.pattern + "errors:", errors)
    fail_init_info = parse_log.parse_test_log(
        f'log/GPT_3_5_Adaptation_{args.pattern}_task0_ctx1_woCoT_api_temp8_log_output.log')
    parse_log.write_failures_to_json(fail_init_info, args.pattern)

    args.pattern = 'test_instruction_4'
    inference = Inference(args)
    inference.pipeline()
    test = Test(args)
    test_results = test.pipeline()
    test.save_results(test_results)
    pass_at_1, overview, errors = test.evaluate_test_results(test_results, 1)
    pass_at_5, _, _ = test.evaluate_test_results(test_results, 5)
    print(args.pattern + "pass@1:", pass_at_1)
    print(args.pattern + "pass@5:", pass_at_5)
    print(args.pattern + "overview:", overview)
    print(args.pattern + "errors:", errors)
    fail_init_info = parse_log.parse_test_log(
        f'log/GPT_3_5_Adaptation_{args.pattern}_task0_ctx1_woCoT_api_temp8_log_output.log')
    parse_log.write_failures_to_json(fail_init_info, args.pattern)

    # for idx in range(3):
    #     args.pattern = 'test_instruction_'+ str(idx)
    #
    #     inference = Inference(args)
    #     inference.pipeline()
    #     test1 = Test(args)
    #     test_results1 = test1.pipeline()
    #     test.save_results(test_results1)
    #     pass_at_1_test, overview, errors = test1.evaluate_test_results(test_results1, 1)
    #     pass_at_5_test, _, _ = test1.evaluate_test_results(test_results1, 5)
    #     print(args.pattern+"pass@1:", pass_at_1_test)
    #     print(args.pattern+"pass@5:", pass_at_5_test)
    #     print(args.pattern+"overview:", overview)
    #     print(args.pattern+"errors:", errors)
    #     fail_init_info1 = parse_log.parse_test_log(
    #         f'log/GPT_3_5_Adaptation_{args.pattern}_task0_ctx1_woCoT_api_temp8_log_output.log')
    #     parse_log.write_failures_to_json(fail_init_info1, args.pattern)
