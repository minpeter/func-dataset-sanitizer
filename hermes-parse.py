from argparse import ArgumentParser
import json, re, ast


parser = ArgumentParser()
parser.add_argument(
    "-f", "--file", "--filename", help="Input file name", dest="filename", required=True
)

parser.add_argument(
    "-d",
    "--debug",
    help="Debug mode",
    dest="debug",
    action="store_true",
)

args = parser.parse_args()


def hermes_system_parser(data, tools_entry):
    try:

        tools_pattern = re.compile(r"<tools>\n(.*?)\n</tools>", re.DOTALL)
        tools_match = tools_pattern.search(data)

        data_string = tools_match.group(1)

        parsed_data = ast.literal_eval(data_string)
        return parsed_data

    except Exception as e:
        try:
            return ast.literal_eval(tools_entry)
        except Exception as e:
            return json.loads(tools_entry)


def tag_list_parser(data, tag):
    parsed_data = []

    tool_call_pattern = re.compile(rf"<{tag}>\n(.*?)\n</{tag}>", re.DOTALL)

    tag_blocks = tool_call_pattern.findall(data)

    for tag_content in tag_blocks:
        try:
            parsed_tag_content = ast.literal_eval(tag_content)
            parsed_data.append(parsed_tag_content)
        except Exception as e:
            parsed_tag_content = json.loads(tag_content)
            parsed_data.append(parsed_tag_content)
    return parsed_data


def parse_function_calling_json(data):
    parsed_data = {
        "messages": [],
        "tools": None,
        "extra": {k: v for k, v in data.items() if k not in ["conversations", "tools"]},
    }

    for conversation in data["conversations"]:
        data_from, data_value = conversation["from"], conversation["value"]

        if data_from == "system":
            # tools 필드가 있는지 확인하고 있다면 넘기기
            if "tools" in data:
                tools_data = hermes_system_parser(data_value, data["tools"])
            else:
                tools_data = hermes_system_parser(data_value, None)
            parsed_data["tools"] = tools_data

        parsed_conversation = {
            "role": data_from,
            "content": data_value,
        }

        if data_from == "human":
            parsed_conversation["role"] = "user"

        if data_from == "gpt":
            parsed_conversation["role"] = "assistant"
            if data_value.startswith("<tool_call>"):

                parse_data = tag_list_parser(data_value, "tool_call")

                tool_calls = []
                for tool_call in parse_data:

                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(
                                    tool_call["arguments"], ensure_ascii=False
                                ),
                            },
                        }
                    )

                parsed_conversation["content"] = tool_calls
            else:
                parsed_conversation["content"] = data_value

        if data_from == "tool":
            parse_data = tag_list_parser(data_value, "tool_response")

            parsed_conversation["content"] = json.dumps(parse_data, ensure_ascii=False)

        if data_from != "system":
            parsed_data["messages"].append(parsed_conversation)

    return parsed_data


def process_jsonl_files(input_file_path, output_file_path, error_file_path):
    """
    Reads a jsonl file, processes each line with parse_function_calling_json,
    and saves the output to a new jsonl file without indentation.
    """

    error_count = 0

    try:
        with open(input_file_path, "r", encoding="utf-8") as infile:
            with open(output_file_path, "w", encoding="utf-8") as outfile:
                with open(error_file_path, "w", encoding="utf-8") as errorfile:
                    for line_num, line in enumerate(infile, 1):
                        try:
                            data = json.loads(line.strip())
                            parsed_data = parse_function_calling_json(data)

                            json.dump(parsed_data, outfile, ensure_ascii=False)
                            outfile.write("\n")
                        except Exception as e:
                            if args.debug:
                                print(f"Error in line {line_num}: {e}")
                            error_count += 1
                            errorfile.write(line)

    except FileNotFoundError:
        print(
            f"Error: File not found at {input_file_path} or {output_file_path} or {error_file_path}"
        )
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")

    total_lines = sum(1 for _ in open(input_file_path, "r", encoding="utf-8"))
    print(
        f"Total lines: {total_lines}, Success: {total_lines - error_count}, Error: {error_count}"
    )


input_file = args.filename
output_file = "./parsed/" + input_file.split(".")[0] + "-parsed.jsonl"
error_file = "./parsed/" + input_file.split(".")[0] + "-error.jsonl"

process_jsonl_files(input_file, output_file, error_file)
