# Original dataset: https://huggingface.co/datasets/cognitivecomputations/dolphin-r1/viewer/reasoning-deepseek/train
# Translated Korean dataset: exp-models/dolphin-r1-korean-deepseek-toolcalls

import json, re, ast
from datasets import Dataset, load_dataset
import pandas as pd


def extract_tools_from_content(content):
    tools_pattern = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
    match = tools_pattern.search(content)
    if not match:
        return None
    tools_str = match.group(1)
    try:
        return ast.literal_eval(tools_str)
    except Exception:
        return json.loads(tools_str)


def extract_tool_calls_from_content(content):
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    match = tool_call_pattern.search(content)
    if not match:
        return None
    tool_calls_str = match.group(1)
    try:
        return ast.literal_eval(tool_calls_str)
    except Exception:
        return json.loads(tool_calls_str)
    
def reasoning_parser(data):
    reasoning_pattern = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
    match = reasoning_pattern.search(data)
    if not match:
        return None
    reasoning_str = match.group(1)
    return reasoning_str.strip() if reasoning_str else None

def parse_function_calling_json(data):
    parsed_data = {
        "messages": [],
        "tools": None,
        "extra": {}
    }
    for conversation in data["messages"]:
        data_role, data_content, data_translated_content = conversation["role"], conversation["content"], conversation.get("translated_content", None)
        
        
        if data_role == "system":
            # print(f"system: {extract_tools_from_content(data_content)}")

            tools = []
            for tool in extract_tools_from_content(data_content):              


                if not isinstance(tool, dict):
                    raise ValueError(f"Tool should be a dictionary, got {type(tool)}: {tool}")
                if "type" in tool and "function" in tool:
                    tools.append(tool)
                    continue
                else:
                    if "name" not in tool or "description" not in tool:
                        raise ValueError(f"Tool must contain 'name' and 'description': {tool}")

                    tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": {
                                "type": "object",
                                "properties": tool.get("parameters", {}),
                                "required": tool.get("required", [])
                            }
                        }
                    })

            parsed_data["tools"] = tools
        elif data_role == "assistant":
            # print(f"reasoning: {reasoning_parser(data_translated_content)}")
            # print(f"assistant: {extract_tool_calls_from_content(data_translated_content)}")

            tool_calls = []
            for tool_call in extract_tool_calls_from_content(data_translated_content):
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
            parsed_data["messages"].append({
                "role": "assistant",
                "tool_calls": tool_calls,
                # "reasoning_content": reasoning_parser(data_translated_content)
            })
        else:
            # print(f"user: {data_translated_content}")
            parsed_data["messages"].append({
                "role": "user",
                "content": data_translated_content
            })
    
    # print(json.dumps(parsed_data, ensure_ascii=False, indent=2))
    return parsed_data

input_ds = load_dataset(
  "exp-models/dolphin-r1-korean-deepseek-toolcalls",
  data_files="data/*.parquet",
)

output = []
error = []

for idx, data in enumerate(input_ds["train"]):
    # for dubugging
    # if idx > 1200:
    #     break

    try:
        parsed = parse_function_calling_json(data)
        output.append(parsed)
    except Exception as e:
        error.append(data)
        print(f"Idx: {idx}, Error: {e}")

# 기존 output_df 생성 및 저장 로직을 row 단위 try-except로 변경

valid_output = []
invalid_indices = []
for idx, row in enumerate(output):
    try:
        # tools 컬럼 처리
        row_copy = row.copy()
        if "tools" not in row_copy:
            row_copy["tools"] = None
        row_copy["tools"] = json.dumps(row_copy["tools"], ensure_ascii=False) if row_copy["tools"] is not None else "null"
        # extra 컬럼 처리
        if "extra" in row_copy and isinstance(row_copy["extra"], dict) and not row_copy["extra"]:
            row_copy["extra"] = None
        valid_output.append(row_copy)
    except Exception as e:
        print(f"Row {idx} 저장 변환 중 에러: {e}")
        invalid_indices.append(idx)
        error.append({"idx": idx, "error": str(e)})

if not valid_output:
    print("Warning: 저장 가능한 row가 없습니다.")
else:
    try:
        output_df = pd.DataFrame(valid_output)
        dataset = Dataset.from_pandas(output_df)
        # output_file_path = f"./parsed/dolphin-r1-korean-deepseek.parquet"
        # dataset.to_parquet(output_file_path)
    except Exception as e:
        print(f"DataFrame을 Dataset으로 변환 중 에러: {e}")
        error.append({"idx": "DataFrame_to_Dataset", "error": str(e)})

output_jsonl_path = f"./parsed/dolphin-r1-korean-deepseek.jsonl"
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for idx, row in enumerate(output):
        if idx in invalid_indices:
            continue
        try:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Row {idx} JSONL 저장 중 에러: {e}")
            error.append({"idx": idx, "error": str(e)})

print(
    f"Total lines: {len(input_ds['train'])}, Success: {len(valid_output)}, Error: {len(error)}"
)
if error:
    print("Error details:")
    for err in error:
        print(f"Idx: {err['idx']}, Error: {err['error']}")
