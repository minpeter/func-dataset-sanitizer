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


def hermes_system_parser(data):
    tools_pattern = re.compile(r"<tools>\n(.*?)\n</tools>", re.DOTALL)
    tools_match = tools_pattern.search(data)

    if not tools_match:
        return None

    data_string = tools_match.group(1)
    parsed_data = ast.literal_eval(data_string)

    return parsed_data


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
        "parsed": [],
        "tools": None,
        "extra": {k: v for k, v in data.items()},
    }

    for conversation in data["conversations"]:
        data_from, data_value = conversation["from"], conversation["value"]
        if data_from == "system":
            tools_data = hermes_system_parser(data_value)
            parsed_data["tools"] = tools_data

        parsed_conversation = {
            "from": conversation["from"],
            "value": conversation["value"],
        }

        if conversation["from"] == "gpt":
            if conversation["value"].startswith("<tool_call>"):
                parsed_conversation["value"] = tag_list_parser(data_value, "tool_call")
            else:
                parsed_conversation["value"] = data_value

        if conversation["from"] == "tool":
            # parsed_conversation["value"] = tag_list_parser(data_value, "tool_response")
            if data_value.startswith("<tool_response>"):
                parsed_conversation["value"] = tag_list_parser(
                    data_value, "tool_response"
                )
            else:
                parsed_conversation["value"] = data_value

        if conversation["from"] != "system":
            parsed_data["parsed"].append(parsed_conversation)

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
