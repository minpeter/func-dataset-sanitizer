import json
from datasets import Dataset, load_dataset
import pandas as pd

def parse_function_calling_json(data):

    parsed = [
        {
            "role": "system",
            "content": data["system"],
            "name": None,
            "tool_calls": None,
        }
    ]

    last_fc_name = None

    for conversation in data["conversations"]:
        from_data, value_data = conversation["from"], conversation["value"]

        parsed_conversation = {
            "role": None,
            "content": None,
            "name": None,
            "tool_calls": None,
        }
        
        if from_data == "human":
            parsed_conversation["role"] = "user"
            parsed_conversation["content"] = value_data
        if from_data == "gpt":
            parsed_conversation["role"] = "assistant"
            parsed_conversation["content"] = value_data
        if from_data == "function_call":
            value_data = json.loads(value_data)
            last_fc_name = value_data["name"]
            parsed_conversation["role"] = "assistant"
            parsed_conversation["content"] = None
            parsed_conversation["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": value_data["name"],
                        "arguments": json.dumps(
                            value_data["arguments"], ensure_ascii=False
                        ),
                    },
                }
            ]
        if from_data == "observation":
            parsed_conversation["role"] = "tool"
            parsed_conversation["name"] = last_fc_name
            parsed_conversation["content"] = value_data

        parsed.append(parsed_conversation)

    parsed_data = {
        "messages": parsed,
        "tools": json.loads(data["tools"]),
        "extra": None,
    }

    return parsed_data


repo = "Salesforce/APIGen-MT-5k"
input_ds = load_dataset(repo)


output = []
error = []

for idx, data in enumerate(input_ds["train"]):
    # # for debugging
    # if idx > 3:
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
output_jsonl_path = f"./parsed/{repo.split('/')[1].lower()}.jsonl"
dataset.to_json(output_jsonl_path, lines=True)

print(
    f"Total lines: {
        len(input_ds['train'])
    }, Success: {len(output)}, Error: {len(error)}"
)
