from argparse import ArgumentParser
import json
import pandas as pd
from datasets import Dataset

args_parser = ArgumentParser()
args_parser.add_argument(
    "-i", "--input", help="Input file name", dest="input", required=True
)
args_parser.add_argument(
    "-a",
    "--answer",
    help="Answer file name",
    dest="answer",
    required=True,
)
args_parser.add_argument(
    "-d",
    "--debug",
    help="Debug mode",
    dest="debug",
    action="store_true",
)
args = args_parser.parse_args()


def modify_data(data_list):
    modified_list = []
    for item in data_list:
        for key, value in item.items():
            arguments = {}
            for arg_key, arg_value in value.items():
                if isinstance(arg_value, list) and arg_value:
                    if isinstance(arg_value[0], dict):
                        arguments[arg_key] = modify_arguments(arg_value[0])
                    elif arg_value[0] == "":
                        continue
                    else:
                        arguments[arg_key] = arg_value[0]
                elif isinstance(arg_value, dict):
                    arguments[arg_key] = modify_arguments(arg_value)
                elif arg_value == "":
                    continue
                else:
                    arguments[arg_key] = arg_value

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
    parsed_list = []
    try:
        with open(input_file_path, "r", encoding="utf-8") as infile:
            with open(answer_file_path, "r", encoding="utf-8") as ansfile:
                for input_line, answer_line in zip(infile, ansfile):
                    try:
                        input_data = json.loads(input_line.strip())
                        answer_data = json.loads(answer_line.strip())

                        if args.debug:
                            print("Input Data:", input_data)
                            print("Answer Data:", answer_data)

                        parsed_data = parse_function_calling_json(
                            input_data, answer_data
                        )
                        if args.debug:
                            print("Parsed Data before DataFrame:", parsed_data)
                        parsed_list.append(parsed_data)

                    except Exception as e:
                        error_count += 1
                        if args.debug:
                            print(f"Error during parsing JSON: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_file_path} or {output_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")

    df = pd.DataFrame(parsed_list)
    df["tools"] = df["tools"].apply(lambda x: json.dumps(x))

    dataset = Dataset.from_pandas(df)

    dataset.to_parquet(output_file_path)

    total_lines = sum(1 for _ in open(input_file_path, "r", encoding="utf-8"))
    print(
        f"Total lines: {total_lines}, Success: {total_lines - error_count}, Error: {error_count}"
    )
    print(f"Output parquet file saved to {output_file_path}")


input_file = args.input
answer_file = args.answer
output_parquet_path = "./parsed/" + input_file.split(".")[0] + ".parquet"

process_jsonl_files(input_file, answer_file, output_parquet_path)
