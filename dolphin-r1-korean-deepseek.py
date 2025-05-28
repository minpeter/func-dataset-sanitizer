# Original dataset: https://huggingface.co/datasets/cognitivecomputations/dolphin-r1/viewer/reasoning-deepseek/train
# Translated Korean dataset: exp-models/dolphin-r1-korean-deepseek-toolcalls
from jsondiff import diff

import json, re, ast
from datasets import Dataset, load_dataset
import pandas as pd


from libs.utils import type2_tool_definition_conv


import logging, os


# 환경변수에서 로깅 레벨 읽기 (없으면 'INFO' 기본값)
loglevel = os.getenv("LOGLEVEL", "INFO").upper()

# 문자열을 실제 로깅 레벨 상수로 변환
numeric_level = getattr(logging, loglevel, logging.INFO)

logging.basicConfig(
    level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


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


def update_array_type1_json_schema(schema):
    """
    Recursively updates a JSON schema dict for type1 tools:
    - If a property is List[str], deduplicate the list and convert to str if only one element remains.
    - If an array has 'prefixItems' and 'type' but no 'items', adds 'items': {}.
    - If an array has neither 'prefixItems' nor 'items', adds 'items': {}.
    """
    if isinstance(schema, dict):
        # First, handle List[str] value deduplication and conversion
        for key, value in list(schema.items()):
            # Check if value is a list of strings
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                # Deduplicate
                deduped = list(dict.fromkeys(value))
                if len(deduped) < len(value):
                    schema[key] = deduped
                # If only one element remains, convert to str
                if len(deduped) == 1:
                    schema[key] = deduped[0]
            elif isinstance(value, dict):
                update_array_type1_json_schema(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        update_array_type1_json_schema(item)

        # Now handle array type schema
        if (
            schema.get("type") == "array"
            or isinstance(schema.get("type"), list)
            and "array" in schema["type"]
        ):
            has_prefix = "prefixItems" in schema
            has_items = "items" in schema
            # If prefixItems exists and items does not, add items: {}
            if has_prefix and not has_items:
                schema["items"] = {}
            # If neither prefixItems nor items, add items: {}
            elif not has_prefix and not has_items:
                schema["items"] = {}
    return schema


def parse_function_calling_json(data):
    parsed_data = {
        "messages": [],
        "tools": None,
        "extra": None,
    }
    for conversation in data["messages"]:
        data_role, data_content, data_translated_content = (
            conversation["role"],
            conversation["content"],
            conversation.get("translated_content", None),
        )

        if data_role == "system":
            # print(f"system: {extract_tools_from_content(data_content)}")

            tools = []
            for tool in extract_tools_from_content(data_content):

                if not isinstance(tool, dict):
                    raise ValueError(
                        f"Tool should be a dictionary, got {type(tool)}: {tool}"
                    )
                if "type" in tool and "function" in tool:
                    # type 1 tool definition

                    # properties의 value 중에서 type이 array인데, items 필드가 없는 경우 items: {} 추가
                    properties = update_array_type1_json_schema(
                        tool["function"]["parameters"].get("properties", {})
                    )

                    remaped_tool = {
                        "type": "function",
                        "function": {
                            "name": tool["function"].get("name", ""),
                            "description": tool["function"].get("description", ""),
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": tool["function"]["parameters"].get(
                                    "required", []
                                ),
                                "additionalProperties": False,
                            },
                        },
                        "strict": True,
                    }
                    # tool과 remaped_tool의 diff를 출력
                    diff_result = diff(tool, remaped_tool)
                    if diff_result:
                        logger.info(
                            f"Tool remapping diff: {diff_result} for tool: {tool}"
                        )
                        # exit(0)
                    tools.append(remaped_tool)
                    continue
                else:
                    # type 2 tool definition
                    if "name" not in tool or "description" not in tool:
                        raise ValueError(
                            f"Tool must contain 'name' and 'description': {tool}"
                        )

                    conv_properties, required = type2_tool_definition_conv(
                        tool.get("parameters", {})
                    )

                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": {
                                    "type": "object",
                                    "properties": conv_properties,
                                    "required": required,
                                },
                            },
                        }
                    )

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
            parsed_data["messages"].append(
                {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                    "reasoning_content": reasoning_parser(data_translated_content),
                }
            )
        else:
            # print(f"user: {data_translated_content}")
            parsed_data["messages"].append(
                {"role": "user", "content": data_translated_content}
            )

    # DEBUG!!!
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
    # if idx > 200:
    #     continue
    # if idx != 10:  # Limit to first 5 for brevity
    #     continue

    try:
        parsed = parse_function_calling_json(data)
        output.append(parsed)
    except Exception as e:
        error.append(data)
        print(f"Idx: {idx}, Error: {e}")


output_df = pd.DataFrame(output)

# output_df에서 1273번 row drop
if 1273 in output_df.index:
    output_df = output_df.drop(index=1273)

# reasoning_content가 포함된 원본 저장
output_df["tools"] = output_df["tools"].apply(
    lambda x: json.dumps(x, ensure_ascii=False)
)
reasoning_ds = Dataset.from_pandas(output_df, preserve_index=False)
output_rfile_path = "./parsed/dolphin-r1-korean-deepseek.parquet"
reasoning_ds.to_parquet(output_rfile_path)


# reasoning_content만 제거한 버전 생성 및 저장
def remove_reasoning_content(messages):
    new_msgs = []
    for m in messages:
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and "reasoning_content" in m
        ):
            m = m.copy()
            m.pop("reasoning_content", None)
        new_msgs.append(m)
    return new_msgs


non_reasoning_df = output_df.copy()
non_reasoning_df["messages"] = non_reasoning_df["messages"].apply(
    remove_reasoning_content
)
non_reasoning_df["tools"] = non_reasoning_df["tools"].apply(
    lambda x: json.dumps(x, ensure_ascii=False)
)
non_reasoning_ds = Dataset.from_pandas(non_reasoning_df)
output_nrfile_path = "./parsed/dolphin-r1-korean-deepseek-non-reasoning.parquet"
non_reasoning_ds.to_parquet(output_nrfile_path)

INPUT_DATASET_LENGTH = len(input_ds["train"])
OUTPUT_DATASET_LENGTH = len(output_df)
print(
    f"Total lines: {INPUT_DATASET_LENGTH}, Saved: {OUTPUT_DATASET_LENGTH}, Error: {INPUT_DATASET_LENGTH - OUTPUT_DATASET_LENGTH}"
)
