# Original dataset: https://huggingface.co/datasets/cognitivecomputations/dolphin-r1/viewer/reasoning-deepseek/train
# Translated Korean dataset: exp-models/dolphin-r1-korean-deepseek-toolcalls

import logging
import os
import json, re, ast
import typing
from datasets import Dataset, load_dataset
import pandas as pd


import logging
from typing import Any, Union, Optional, Dict, List
from pydantic import TypeAdapter



# 환경변수에서 로깅 레벨 읽기 (없으면 'INFO' 기본값)
loglevel = os.getenv("LOGLEVEL", "INFO").upper()

# 문자열을 실제 로깅 레벨 상수로 변환
numeric_level = getattr(logging, loglevel, logging.INFO)

logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def python_type_to_json_schema(python_type: str, test: str) -> dict:
    """
    Converts a Python type string (like 'List[int]', 'Optional[Dict[str, float]]', etc.)
    to a full JSON Schema (as a dict).
    Covers common broken/partial/unsupported types gracefully.
    """
    # Normalize and patch common broken types
    type_map = {
        "list": "List[Any]",
        "set": "List[Any]",  # JSON Schema does not support set, treat as list
        "tuple": "List[Any]",  # JSON Schema does not support tuple, treat as list
        "Tuple": "List[Any]",
        "List": "List[Any]",
        "dict": "Dict[str, Any]",
        "Callable": None,  # Not supported in JSON Schema
        "Any": "Any",
    }

    # Remove trailing incomplete brackets (e.g., Tuple[float)
    def fix_brackets(tp: str) -> str:
        # If there are more '[' than ']', add missing ']'
        n_open = tp.count('[')
        n_close = tp.count(']')
        if n_open > n_close:
            tp = tp + (']' * (n_open - n_close))
        return tp

    # Remove whitespace and handle common cases
    tp = python_type.strip()
    tp = tp.replace(" ", "")
    # Remove trailing commas (e.g., Tuple[float, float,])
    if tp.endswith(",]"):
        tp = tp.replace(",]", "]")
    # Patch for known broken types
    if tp in type_map:
        tp = type_map[tp]
        if tp is None:
            logger.info(f"Type '{python_type}' is not supported for JSON schema.")
            return None
    # Patch for Tuple[...] -> List[Any]
    if tp.startswith("Tuple["):
        tp = "List[Any]"
    # Patch for List[Tuple...] -> List[List[Any]]
    if tp.startswith("List[Tuple"):
        tp = "List[List[Any]]"
    # Patch for List[Union[int... (broken bracket)
    if tp.startswith("List[Union["):
        tp = "List[Any]"
    # Patch for List[...] with broken bracket
    tp = fix_brackets(tp)
    # Patch for Callable[...] (not supported)
    if tp.startswith("Callable"):
        logger.info(f"Type '{python_type}' is not supported for JSON schema.")
        return {}
    # Special case: broken Callable like 'Callable[[float]'
    if tp.startswith("Callable[["):
        logger.info(f"Type '{python_type}' is not supported for JSON schema.")
        return {}
    if tp.startswith("List[Union["):
        tp = "List[Any]"
    # Patch for List[...] with broken bracket
    tp = fix_brackets(tp)
    # Patch for Callable[...] (not supported)
    if tp.startswith("Callable"):
        logger.info(f"Type '{python_type}' is not supported for JSON schema.")
        return {}

    # 안전한 eval 환경
    eval_env = {
        "List": list,
        "Dict": dict,
        "Union": typing.Union,
        "Optional": typing.Optional,
        "Any": Any,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "NoneType": type(None),
        "None": type(None),
        "__builtins__": {},
    }

    try:
        py_type = eval(tp, eval_env)
    except Exception as e:
        logger.warning(f"Type parsing error for '{python_type}': {str(e)}")
        logger.warning(f"Test value: {test}")
        return {}

    try:
        schema = TypeAdapter(py_type).json_schema()
        return schema
    except Exception as e:
        logger.warning(f"Schema generation error for '{python_type}': {str(e)}")
        return {}


def parse_type(type_str: str):
    # 안전한 eval 환경
    allowed = {
        "list": list,
        "List": list,
        "Dict": dict,
        "Union": typing.Union,
        "Optional": typing.Optional,
        "Any": Any,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "None": type(None),
        "NoneType": type(None),
    }
    try:
        return eval(type_str, {"__builtins__": {}}, allowed)
    except Exception as e:
        raise ValueError(f"Type parsing error: {type_str} ({e})")

# 값과 타입 객체를 받아서 재귀적으로 캐스팅
def cast_value(value, type_obj):
    origin = typing.get_origin(type_obj)
    args = typing.get_args(type_obj)

    # 기본 타입
    if type_obj in [str, int, float, bool]:
        try:
            if type_obj is bool:
                if isinstance(value, str):
                    if value.lower() == "true":
                        return True
                    elif value.lower() == "false":
                        return False
                return bool(value)
            return type_obj(value)
        except Exception:
            return type_obj()  # 기본값 반환

    # NoneType
    if type_obj is type(None):
        return None

    # Optional[T] == Union[T, NoneType]
    if origin is typing.Union and type(None) in args:
        other_type = args[0] if args[1] is type(None) else args[1]
        if value is None or (isinstance(value, str) and value.lower() == "none"):
            return None
        return cast_value(value, other_type)

    # Union
    if origin is typing.Union:
        for arg in args:
            try:
                return cast_value(value, arg)
            except Exception:
                continue
        # 모두 실패하면 첫 번째 타입의 기본값
        return cast_value(None, args[0])

    # List
    if origin is list or origin is typing.List:
        if not isinstance(value, (list, tuple)):
            # 콤마로 구분된 str도 리스트로 변환
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",")]
            else:
                value = [value]
        elem_type = args[0] if args else Any
        return [cast_value(v, elem_type) for v in value]

    # Dict
    if origin is dict or origin is typing.Dict:
        key_type = args[0] if args else Any
        val_type = args[1] if len(args) > 1 else Any
        if not isinstance(value, dict):
            return {}
        return {cast_value(k, key_type): cast_value(v, val_type) for k, v in value.items()}

    # Any
    if type_obj is Any:
        return value

    # fallback
    return value

# 통합 함수: 타입 문자열과 값을 받아서 캐스팅
def cast_with_type_str(value, type_str: str):

    if not type_str:
        # 타입 문자열이 비어있으면 기본값으로 처리
        return None

    try:
        type_obj = parse_type(type_str)
    except Exception as e:
        print(f"[Type Parse Error] {e}")
        return value
    try:
        return cast_value(value, type_obj)
    except Exception as e:
        print(f"[Cast Error] {e}")
        return value


def type2_tool_definition_conv(parameters: dict):
    """
    Converts a type2 tool definition's parameters:
    - Splits the 'type' field by ',' and maps the first type to Python type using json_to_python_type.
    - If 'optional' is not present in the type, adds the key to required list.
    Returns (new_properties_dict, required_list).
    """
    new_properties = {}
    required = []
    for key, value in parameters.items():
        type_str = value.get("type", "")
        type_parts = [t.strip() for t in type_str.split(",")]
        base_type = type_parts[0] if type_parts else "any"

        json_schema = python_type_to_json_schema(base_type, value.get("type", ""))

        # logger.info(f"Python type for {key}: {base_type}")
        # logger.info(f"JSON schema for {key}: {json_schema}")

        new_value = json_schema.copy() if json_schema is not None else {}
        if "description" in value:
            new_value["description"] = value["description"]

        if "default" in value and json_schema is not None:
            cast_value = cast_with_type_str(value["default"], base_type)
            if type(cast_value) != type(value["default"]):
                logger.info(f"Default value for {key} changed from {type(value['default'])} to {type(cast_value)} due to type casting.")
                logger.info(f"Original value: {value['default']}, Casted value: {cast_value}")
                if str(cast_value) != str(value["default"]):
                    logger.warning(f"뭔가 사람이 보기에 값이 달라진거 같다고 생각함, PASS")
                elif cast_value is not None:
                    logger.info("적절해보임. 케스팅되었고, default 값을 설정함.")
                    new_value["default"] = cast_value
            else:
                logger.info("케스팅 전과 후가 완전히 동일함. default 값을 설정함.")
                new_value["default"] = value["default"]

        new_properties[key] = new_value
        if not any("optional" == t for t in type_parts[1:]):
            required.append(key)
    return new_properties, required

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
        "extra": None,
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
                    # type 1 tool definition
                    tools.append(tool)
                    continue
                else:
                    # type 2 tool definition
                    if "name" not in tool or "description" not in tool:
                        raise ValueError(f"Tool must contain 'name' and 'description': {tool}")
                    
                    conv_properties, required = type2_tool_definition_conv(tool.get("parameters", {}))

                    tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": {
                                "type": "object",
                                "properties": conv_properties,
                                "required": required
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

# for debugging
# print(output_df.iloc[0].to_json(indent=2))

# Since each tool has different properties, convert to string to meet the requirements of parquet.
output_df["tools"] = output_df["tools"].apply(lambda x: json.dumps(x))

# __index_level_0__ 컬럼 제거
if "__index_level_0__" in output_df.columns:
    output_df = output_df.drop(columns="__index_level_0__")

dataset = Dataset.from_pandas(output_df)

output_file_path = f"./parsed/dolphin-r1-korean-deepseek-non-reasoning.parquet"
dataset.to_parquet(output_file_path)

print(
    f"Total lines: {
    len(input_ds['train'])
    }, Success: {len(output)}, Error: {len(error)}"
)

# 기존 output_df 생성 및 저장 로직을 row 단위 try-except로 변경

# valid_output = []
# invalid_indices = []
# for idx, row in enumerate(output):
#     try:
#         # tools 컬럼 처리
#         row_copy = row.copy()
#         if "tools" not in row_copy:
#             row_copy["tools"] = None
#         row_copy["tools"] = json.dumps(row_copy["tools"], ensure_ascii=False) if row_copy["tools"] is not None else "null"
#         # extra 컬럼 처리
#         if "extra" in row_copy and isinstance(row_copy["extra"], dict) and not row_copy["extra"]:
#             row_copy["extra"] = None
#         valid_output.append(row_copy)
#     except Exception as e:
#         print(f"Row {idx} 저장 변환 중 에러: {e}")
#         invalid_indices.append(idx)
#         error.append({"idx": idx, "error": str(e)})

# if not valid_output:
#     print("Warning: 저장 가능한 row가 없습니다.")
# else:
#     try:
#         output_df = pd.DataFrame(valid_output)
#         dataset = Dataset.from_pandas(output_df)
#         # output_file_path = f"./parsed/dolphin-r1-korean-deepseek.parquet"
#         # dataset.to_parquet(output_file_path)
#     except Exception as e:
#         print(f"DataFrame을 Dataset으로 변환 중 에러: {e}")
#         error.append({"idx": "DataFrame_to_Dataset", "error": str(e)})

# output_jsonl_path = f"./parsed/dolphin-r1-korean-deepseek.jsonl"
# output_non_reasoning_jsonl_path = f"./parsed/dolphin-r1-korean-deepseek-non-reasoning.jsonl"

# with open(output_jsonl_path, "w", encoding="utf-8") as f_reasoning, \
#      open(output_non_reasoning_jsonl_path, "w", encoding="utf-8") as f_non_reasoning:
#     for idx, row in enumerate(output):
#         if idx in invalid_indices:
#             continue
#         try:
#             # 저장: reasoning 포함
#             f_reasoning.write(json.dumps(row, ensure_ascii=False) + "\n")
#             # 저장: reasoning 미포함 (reasoning_content 필드 제거)
#             row_non_reasoning = json.loads(json.dumps(row, ensure_ascii=False))
#             if "messages" in row_non_reasoning:
#                 for msg in row_non_reasoning["messages"]:
#                     if "reasoning_content" in msg:
#                         del msg["reasoning_content"]
#             f_non_reasoning.write(json.dumps(row_non_reasoning, ensure_ascii=False) + "\n")
#         except Exception as e:
#             print(f"Row {idx} JSONL 저장 중 에러: {e}")
#             error.append({"idx": idx, "error": str(e)})

# print(
#     f"Total lines: {len(input_ds['train'])}, Success: {len(valid_output)}, Error: {len(error)}"
# )
# if error:
#     print("Error details:")
#     for err in error:
#         print(f"Idx: {err['idx']}, Error: {err['error']}")
