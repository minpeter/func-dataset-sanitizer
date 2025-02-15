from argparse import ArgumentParser
import json


parser = ArgumentParser()
parser.add_argument(
    "-i", "--input", "--inputfile", help="Input file name", dest="input", required=True
)

parser.add_argument(
    "-a",
    "--answer",
    "--answerfile",
    help="Answer file name",
    dest="answer",
    required=True,
)

parser.add_argument(
    "-d",
    "--debug",
    help="Debug mode",
    dest="debug",
    action="store_true",
)

args = parser.parse_args()


def modify_data(data_list):
    """
    입력받은 데이터를 특정 형태로 수정하는 함수입니다.

    Args:
        data_list: 수정할 데이터 리스트. 각 요소는 딕셔너리 형태여야 합니다.

    Returns:
        수정된 데이터 리스트. 각 요소는 {"name": "", "arguments": {}} 형태의 딕셔너리입니다.
    """
    modified_list = []
    for item in data_list:
        for key, value in item.items():
            arguments = {}
            for arg_key, arg_value in value.items():
                # 여러번 타고 내려가는 경우를 고려하여 재귀적으로 처리
                if (
                    isinstance(arg_value, list) and arg_value
                ):  # 리스트이고 비어있지 않은 경우 첫번째 요소 사용
                    if isinstance(
                        arg_value[0], dict
                    ):  # 리스트의 첫번째 요소가 딕셔너리인 경우, 다시 재귀적으로 처리 (하지만 이 문제에서는 필요 없을듯)
                        arguments[arg_key] = modify_arguments(
                            arg_value[0]
                        )  # 필요하다면 재귀 호출, 현재 문제에서는 불필요해 보임.
                    elif arg_value[0] == "":
                        continue  # 빈 문자열인 경우 무시
                    else:
                        arguments[arg_key] = arg_value[0]  # 리스트의 첫번째 요소 사용
                elif isinstance(
                    arg_value, dict
                ):  # 딕셔너리인 경우, 재귀적으로 처리 (하지만 이 문제에서는 필요 없을듯)
                    arguments[arg_key] = modify_arguments(
                        arg_value
                    )  # 필요하다면 재귀 호출, 현재 문제에서는 불필요해 보임.
                elif arg_value == "":
                    continue  # 빈 문자열인 경우 무시
                else:
                    arguments[arg_key] = arg_value  # 리스트가 아니면 그대로 사용

            modified_list.append(
                {
                    "type": "function",
                    "function": {
                        "name": key,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
    return modified_list


def modify_arguments(arg_dict):
    """
    arguments 딕셔너리를 재귀적으로 처리하는 함수

    Args:
        arg_dict: arguments 딕셔너리

    Returns:
        수정된 arguments 딕셔너리
    """
    modified_args = {}
    for arg_key, arg_value in arg_dict.items():
        if isinstance(arg_value, list) and arg_value:
            if isinstance(arg_value[0], dict):
                modified_args[arg_key] = modify_arguments(arg_value[0])
            else:
                modified_args[arg_key] = arg_value[0]
        elif isinstance(arg_value, dict):
            modified_args[arg_key] = modify_arguments(arg_value)
        else:
            modified_args[arg_key] = arg_value
    return modified_args


def parse_function_calling_json(input_data, answer_data):
    if input_data["id"] != answer_data["id"]:
        raise ValueError("ID mismatch")

    # Parse function definitions from input_data["functionc"]
    function_defs = []
    for func in input_data["function"]:
        function_defs.append({"type": "function", "function": func})

    parsed_data = {
        "messages": [
            {
                "role": "user",
                "content": input_data["question"][0][0]["content"],
            },
            {
                "role": "assistant",
                "tool_calls": modify_data(answer_data["ground_truth"]),
            },
        ],
        "tools": function_defs,
        "extra": {
            "id": input_data["id"],
        },
    }

    return parsed_data


def process_jsonl_files(input_file_path, answer_file_path, output_file_path):

    error_count = 0
    try:
        with open(input_file_path, "r", encoding="utf-8") as infile:
            with open(answer_file_path, "r", encoding="utf-8") as ansfile:
                with open(output_file_path, "w", encoding="utf-8") as outfile:
                    for input_line, answer_line in zip(infile, ansfile):
                        try:
                            input_data = json.loads(input_line.strip())
                            answer_data = json.loads(answer_line.strip())

                            parsed_data = parse_function_calling_json(
                                input_data, answer_data
                            )

                            json.dump(parsed_data, outfile, ensure_ascii=False)
                            outfile.write("\n")
                        except Exception as e:
                            error_count += 1
                            if args.debug:
                                print(f"Error: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_file_path} or {output_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")

    total_lines = sum(1 for _ in open(input_file_path, "r", encoding="utf-8"))
    print(
        f"Total lines: {total_lines}, Success: {total_lines - error_count}, Error: {error_count}"
    )


input_file = args.input
answer_file = args.answer

output_file = "./parsed/" + input_file.split(".")[0] + "-parsed.jsonl"

process_jsonl_files(input_file, answer_file, output_file)
