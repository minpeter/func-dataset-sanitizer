import json, re, ast
from datasets import Dataset, load_dataset
import pandas as pd


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

    if "id" in data and data["id"] == "e39781a4-29a3-4896-83d2-0ab5d264b6ed":
        raise Exception("Skip this data")

    for conversation in data["conversations"]:
        data_from, data_value = conversation["from"], conversation["value"]

        if data_from == "system":
            if "tools" in data:
                # handle glaive function dataset
                tools_data = hermes_system_parser(data_value, data["tools"])
            else:
                tools_data = hermes_system_parser(data_value, None)

            parsed_data["tools"] = tools_data

        parsed_conversation = {
            "role": None,
            "content": None,
            "tool_calls": None,
        }

        if data_from == "human":
            parsed_conversation["role"] = "user"
            parsed_conversation["content"] = data_value

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

                parsed_conversation["tool_calls"] = tool_calls
            else:
                parsed_conversation["content"] = data_value

        if data_from == "tool":
            parse_data = tag_list_parser(data_value, "tool_response")
            for tool in parse_data:

                if "name" in tool and "content" in tool:
                    parsed_data["messages"].append(
                        {
                            "role": "tool",
                            "name": tool["name"],
                            "content": json.dumps(tool["content"], ensure_ascii=False),
                        }
                    )
                else:
                    parsed_data["messages"].append(
                        {
                            "role": "tool",
                            "content": json.dumps(tool, ensure_ascii=False),
                        }
                    )
        elif data_from != "system":
            parsed_data["messages"].append(parsed_conversation)

    return parsed_data


target_files = [
    "func-calling.json",
    "func-calling-singleturn.json",
    "glaive-function-calling-5k.json",
]

for target_file in target_files:
    input_ds = load_dataset(
        "NousResearch/hermes-function-calling-v1",
        data_files={
            "train": [
                target_file,
            ]
        },
    )

    output = []
    error = []

    for idx, data in enumerate(input_ds["train"]):
        try:
            output.append(parse_function_calling_json(data))
        except Exception as e:
            error.append(data)
            print(f"Idx: {idx}, Error: {e}")

    output_df = pd.DataFrame(output)

    # for debugging
    # print(output_df.iloc[0].to_json(indent=2))

    # Since each tool has different properties, convert to string to meet the requirements of parquet.
    output_df["tools"] = output_df["tools"].apply(lambda x: json.dumps(x))

    dataset = Dataset.from_pandas(output_df)

    output_file_path = f"./parsed/{target_file.split('.')[0]}.parquet"
    dataset.to_parquet(output_file_path)

    print(
        f"Total lines: {
            len(input_ds['train'])
        }, Success: {len(output)}, Error: {len(error)}"
    )
