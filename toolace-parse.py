import json, re
from datasets import Dataset, load_dataset
import pandas as pd
from sympy import N


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

    tools = []

    for tool in parsed_data:

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
        )

    return tools


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
        raise ValueError("Invalid API list text format")

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

        from_data, value_data = conversation["from"], conversation["value"]

        parsed_conversation = {
            "role": from_data,
            "content": value_data,
        }

        if from_data == "assistant":
            if value_data.startswith("["):

                tool_parse = parse_api_list_text_to_json_list(value_data)

                tool_calls = []

                for tool in tool_parse:
                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "arguments": json.dumps(
                                    tool["arguments"], ensure_ascii=False
                                ),
                            },
                        }
                    )

                parsed_conversation["tool_calls"] = tool_calls
                parsed_conversation["content"] = None
            else:
                parsed_conversation["content"] = value_data

        if from_data == "tool":
            for calls in json.loads(value_data):
                parsed.append(
                    {
                        "role": "tool",
                        "name": calls["name"],
                        "content": json.dumps(calls["results"], ensure_ascii=False),
                    }
                )
        else:
            parsed.append(parsed_conversation)

    parsed_data = {
        "messages": parsed,
        "tools": toolace_system_parser(data["system"]),
        "extra": None,
    }

    return parsed_data


repo = "Team-ACE/ToolACE"
input_ds = load_dataset(repo)


output = []
error = []

for idx, data in enumerate(input_ds["train"]):
    # for debugging
    # if idx > 0:
    #     break
    try:
        output.append(parse_function_calling_json(data))
    except Exception as e:
        error.append(data)
        print(f"Idx: {idx}, Error: {e}")

output_df = pd.DataFrame(output)

# Since each tool has different properties, convert to string to meet the requirements of parquet.
output_df["tools"] = output_df["tools"].apply(lambda x: json.dumps(x))

# for debugging
# print(output_df.iloc[0].to_json(indent=2))

dataset = Dataset.from_pandas(output_df)

output_file_path = f"./parsed/{repo.split('/')[1].lower()}.parquet"
dataset.to_parquet(output_file_path)

print(
    f"Total lines: {
        len(input_ds['train'])
    }, Success: {len(output)}, Error: {len(error)}"
)
