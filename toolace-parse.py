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


def toolace_system_parser(data):
    tools_pattern = re.compile(
        r"\[{(.*?)}\]",
        re.DOTALL,
    )
    tools_match = tools_pattern.search(data)

    if not tools_match:
        return None

    data_string = tools_match.group(1)
    data_string = "[{" + data_string + "}]"

    # print(data_string)

    parsed_data = json.loads(data_string)

    return parsed_data


def parse_api_text_to_json(api_text):
    """
    아래 형식의 텍스트를 JSON으로 파싱합니다.
    예: "Market Trends API(trend_type=\"MARKET_INDEXES\", country=\"us\")"

    Args:
        api_text (str): 파싱할 텍스트

    Returns:
        dict: JSON 형식으로 파싱된 데이터
    """
    api_text = api_text.strip()
    if api_text.startswith("["):  # remove bracket if exists at the beginning
        api_text = api_text[1:]
    if api_text.endswith("]"):  # remove bracket if exists at the end
        api_text = api_text[:-1]

    # API 이름과 인수를 분리하기 위해 정규 표현식 사용
    match = re.match(r"([^()]+)\((.*)\)", api_text)
    if not match:
        # 인수가 없는 경우를 처리 (예: "Simple API")
        api_name = api_text.strip()
        return {"name": api_name, "arguments": {}}

    api_name = match.group(1).strip()
    arguments_str = match.group(2).strip()

    arguments = {}
    if arguments_str:
        # 쉼표로 인수를 분리하고 각 인수를 파싱
        for arg_pair in arguments_str.split(","):
            arg_pair = arg_pair.strip()
            if "=" in arg_pair:
                key, value = arg_pair.split("=", 1)
                arguments[key.strip()] = value.strip().strip('"')  # 따옴표 제거

    return {"name": api_name, "arguments": arguments}


def parse_api_list_text_to_json_list(api_list_text):
    """
    아래 형식의 텍스트 리스트를 JSON 객체 리스트로 파싱합니다.
    예: "[API_CALL_1, API_CALL_2, API_CALL_3]"

    Args:
        api_list_text (str): 파싱할 API 호출 리스트 텍스트

    Returns:
        list: JSON 객체 리스트
    """
    api_list_text = api_list_text.strip()
    if not api_list_text.startswith("[") or not api_list_text.endswith("]"):
        print(
            "경고: 입력 텍스트가 '['와 ']'로 시작하고 끝나지 않습니다. 전체 텍스트를 하나의 API 호출로 처리합니다."
        )
        # 괄호가 없으면, 전체 텍스트를 하나의 API 호출로 처리 시도 (오류를 최대한 방지)
        parsed_api = parse_api_text_to_json(api_list_text)
        return [parsed_api] if parsed_api else []  # 파싱 실패 시 빈 리스트 반환

    api_list_text = api_list_text[1:-1]  # Remove brackets
    api_texts = api_list_text.split(
        "), "
    )  # Split by '), ' which separates API calls, but be aware of the last element

    json_list = []
    for api_text_with_potential_ending_comma in api_texts:
        api_text = api_text_with_potential_ending_comma  # initially assume it is the full api_text
        if not api_text.endswith(
            ")"
        ):  # it is not the last element and still needs to be completed by adding ')'
            api_text += ")"  # re-attach ')'

        parsed_json = parse_api_text_to_json(api_text)
        if parsed_json:  # Check if parsing was successful
            json_list.append(parsed_json)

    return json_list


def parse_function_calling_json(data):

    parsed = []

    for conversation in data["conversations"]:

        parsed_conversation = {
            "from": conversation["from"],
            "value": conversation["value"],
        }

        if conversation["from"] == "assistant":
            if conversation["value"].startswith("["):
                parsed_conversation["value"] = parse_api_list_text_to_json_list(
                    conversation["value"]
                )
            else:
                parsed_conversation["value"] = conversation["value"]

        if conversation["from"] == "tool":
            parsed_conversation["value"] = json.loads(conversation["value"])

        parsed.append(parsed_conversation)

    parsed_data = {
        "parsed": parsed,
        "tools": toolace_system_parser(data["system"]),
        "extra": {k: v for k, v in data.items()},
    }

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
