# for MadeAgents/xlam-irrelevance-7.5k dataset sanitization

import os, json
from openai import OpenAI
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re

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

    tools_list = json.loads(data["tools"])
    system_prompt = f"""Here is the list of tools available to you:
<tools>
{data["tools"]}
</tools>

You are an advanced AI assistant with expertise in various domains and access to specialized tools. Your primary function is to assist users by either utilizing these tools or providing information from your general knowledge base.

When interacting with users, follow these guidelines:

1. Analyze the user's query carefully.
2. Determine if the query is related to any of the available tools.
3. If the query is tool-related:
   - Identify the most appropriate tool.
   - Format the tool call exactly as specified below.
4. If the query is not tool-related:
   - For general or conversational queries, respond naturally using your internal knowledge.
   - For queries about specific domains not covered by the tools, explain why you can't process the request.

Tool Call Format:
When calling a tool, use this exact format:
```tool_call
[tool_name(param=value, param2=value2)]
```
Ensure that "```tool_call" precedes every tool call.

Before responding to each query, analyze the situation and determine the best course of action inside <analysis> tags within your thinking block. Your analysis should include:

1. Query analysis: Summarize the user's request.
2. Tool relevance: List each available tool and briefly note its relevance to the query.
3. If a tool seems relevant:
   - Note each required parameter for the tool.
   - Check if the parameter information is present in the user's query.
4. If no tool is relevant:
   - Determine if this is a general knowledge query or a request for unavailable information.
5. Response approach: Outline how you plan to respond, whether using a tool or general knowledge.

Pay special attention to crafting natural, human-like responses for queries unrelated to the tools.

Example thought process:
<analysis>
1. Query analysis: The user is asking about the weather in New York.
2. Tool relevance:
   - Weather tool: Not available in the tool list.
   - [List other tools]: Not relevant to weather queries.
3. Tool use: No relevant tool available.
4. Query type: This is a general information request.
5. Response approach: I'll provide a friendly, conversational response using my general knowledge about weather patterns and New York.
</analysis>

Remember, your goal is to be helpful, informative, and engaging in your responses, whether you're using a tool or not. Your final output should consist only of your response or tool call, and should not duplicate or rehash any of the work you did in the analysis section."""

    response = client.chat.completions.create(
        model=model_id,
        n=1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data["query"]},
        ],
    )

    llm_resp = response.choices[0].message.content

    # Extract analysis content between <analysis> tags
    analysis_match = re.search(r"<analysis>(.*?)</analysis>", llm_resp, re.DOTALL)
    analysis_resp = analysis_match.group(1).strip() if analysis_match else ""

    # Get plain response by removing analysis section
    plain_resp = re.sub(
        r"<analysis>.*?</analysis>", "", llm_resp, flags=re.DOTALL
    ).strip()

    if "tool_call" in plain_resp:
        raise Exception("Failed to generate tool calls")

    tools = []
    for tool in tools_list:
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

    parsed_data = {
        "messages": [
            {
                "role": "user",
                "content": data["query"],
            },
            {"role": "assistant", "content": plain_resp},
        ],
        "tools": tools,
        "extra": {
            "distill": "Qwen2.5-72B-Instruct",
            "analysis": analysis_resp,
        },
    }

    return parsed_data


repo = "MadeAgents/xlam-irrelevance-7.5k"
input_ds = load_dataset(repo)


def process_item(data):
    if len(json.loads(data["tools"])) == 0:
        print("No tools available")
        return {"success": False, "data": data, "error": "No tools available"}
    try:
        return {"success": True, "data": parse_function_calling_json(data)}
    except Exception as e:
        print(f"Error: {str(e)}")
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

print(output_df.head())

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
