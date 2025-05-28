import json
from datasets import Dataset, load_dataset
import pandas as pd
from libs.utils import type2_tool_definition_conv


def parse_function_calling_json(data):

    answers = json.loads(data["answers"])
    tool_calls = []
    for answer in answers:
        tool_calls.append(
            {
                "type": "function",
                "function": {
                    "name": answer["name"],
                    "arguments": json.dumps(answer["arguments"], ensure_ascii=False),
                },
            }
        )

    tools = []
    for tool in json.loads(data["tools"]):

        conv_properties, required = type2_tool_definition_conv(tool["parameters"])
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "parameters": {
                        "type": "object",
                        "properties": conv_properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )

    parsed_data = {
        "messages": [
            {
                "role": "user",
                "content": data["query"],
            },
            {"role": "assistant", "tool_calls": tool_calls},
        ],
        "tools": tools,
        "extra": {
            "id": data["id"],
        },
    }

    return parsed_data


repo = "Salesforce/xlam-function-calling-60k"
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

output_file_path = f"./parsed/{repo.split('/')[1]}.parquet"
dataset.to_parquet(output_file_path)

print(
    f"Total lines: {
        len(input_ds['train'])
    }, Success: {len(output)}, Error: {len(error)}"
)
