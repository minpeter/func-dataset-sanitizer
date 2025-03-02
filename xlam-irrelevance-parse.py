# for MadeAgents/xlam-irrelevance-7.5k dataset sanitization

import os, json
from openai import OpenAI
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

load_dotenv()

# client = OpenAI(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )
# model_id = "gemini-2.0-flash-thinking-exp-01-21"

# client = OpenAI(
#     api_key=os.getenv("FRIENDLI_TOKEN"),
#     base_url="https://api.friendli.ai/serverless/v1/",
# )
# model_id = "meta-llama-3.1-8b-instruct"


client = OpenAI(
    api_key=os.getenv("FRIENDLI_TOKEN"),
    base_url="https://api.friendli.ai/dedicated/v1/",
)
model_id = os.getenv("FRIENDLI_EID")


def parse_function_calling_json(data):
    system_prompt = """A dataset of tools that are not related to the user's query is provided one line at a time. First, consider whether the user's question is really not related to the given tool, and if not, respond that it cannot be processed. This is a task to fill the response of the unrelated dataset.
If some of the user's requests can be processed by calling a function and some cannot be processed, respond that some tasks can be processed but the entire request cannot be processed instead of calling a function.
    -----
You are an expert in function calling.
Given a question and a set of possible tools, decide if a tool should be invoked.

Format tool calls strictly as:
```tool_call
[tool_name(param=value, param2=value2)]
```
**Important:** Tool calls must begin with "```tool_call".

If no tool is relevant or required parameters are missing, state that you cannot process the request. Do not use insider knowledge or hallucinate.  If you decline, briefly explain why.
Available tools: """ + json.dumps(
        data["tools"], indent=4
    )

    tools = json.loads(data["tools"])

    if not data["tools"]:
        response = client.chat.completions.create(
            model=model_id,
            n=1,
            messages=[
                {"role": "user", "content": data["query"]},
            ],
        )
    else:
        response = client.chat.completions.create(
            model=model_id,
            n=1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data["query"]},
            ],
        )

    tools = []
    for tool in tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "parameters": {
                        "type": "object",
                        "properties": tool["parameters"],
                    },
                },
            }
        )

    llm_resp = response.choices[0].message.content

    if "tool_call" in llm_resp:
        raise Exception("Failed to generate tool calls")

    parsed_data = {
        "messages": [
            {
                "role": "user",
                "content": data["query"],
            },
            {"role": "assistant", "content": llm_resp},
        ],
        "tools": tools,
        "extra": {
            "distill": "Qwen2.5-72B-Instruct",
        },
    }

    return parsed_data


repo = "MadeAgents/xlam-irrelevance-7.5k"
input_ds = load_dataset(repo)


def process_item(data):
    try:
        return {"success": True, "data": parse_function_calling_json(data)}
    except Exception as e:
        return {"success": False, "data": data, "error": str(e)}


num_processes = cpu_count()
with Pool(processes=num_processes) as pool:
    results = list(
        tqdm(pool.imap(process_item, input_ds["train"]), total=len(input_ds["train"]))
    )

output = []
error = []

for idx, result in enumerate(results):
    if result["success"]:
        output.append(result["data"])
    else:
        error.append(result["data"])
        print(f"Idx: {idx}, Error: {result['error']}")

output_df = pd.DataFrame(output)
output_df["tools"] = output_df["tools"].apply(lambda x: json.dumps(x))

dataset = Dataset.from_pandas(output_df)
error_df = pd.DataFrame(error)

output_file_path = f"./parsed/{repo.split('/')[1]}.parquet"
output_file_path_jsonl = f"./parsed/{repo.split('/')[1]}.jsonl"
dataset.to_parquet(output_file_path)
dataset.to_json(output_file_path_jsonl)

error_file_path = f"./parsed/{repo.split('/')[1]}-error.parquet"
error_file_path_jsonl = f"./parsed/{repo.split('/')[1]}-error.jsonl"
error_df.to_parquet(error_file_path)
error_df.to_json(error_file_path_jsonl, orient="records", lines=True)

print(f"Total lines: {len(input_ds['train'])}")
print(f"Success: {len(output)}")
print(f"Error: {len(error)}")
