from jsondiff import diff

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

logging.basicConfig(
    level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def python_type_to_json_schema(python_type: str, test: str) -> dict:
    """
    Converts a Python type string (like 'List[int]', 'Optional[Dict[str, float]]', etc.)
    to a full JSON Schema (as a dict).
    Covers common broken/partial/unsupported types gracefully.
    Ensures that for list/array types, 'items': {} is always present.
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
        n_open = tp.count("[")
        n_close = tp.count("]")
        if n_open > n_close:
            tp = tp + ("]" * (n_open - n_close))
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
        # 반드시 array 타입이면 items: {} 포함
        if schema.get("type") == "array":
            if "items" not in schema or not schema["items"]:
                schema["items"] = {}
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
        return {
            cast_value(k, key_type): cast_value(v, val_type) for k, v in value.items()
        }

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


# 여기서 type2란, xlam에서 포맷팅한 파이썬 기반 tool parameters를 의미함.
# 대표적인 특징으로는 "type" 필드에 string대신 str같은 파이썬 타입이 들어가고,
# required 필드가 별도로 존재하는 대신 type 필드에 type: <type>, optional 이런식으로 표시된 type이 아니면 required로 간주한다.

# 현재로썬 dolphin-r1-toolcall과 xlam에서 공유한다. (추측하건데 dolphin-r1 toolcall의 데이터 중 일부가 xlam의 믹스일 가능성이 있다.)


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
                logger.info(
                    f"Default value for {key} changed from {type(value['default'])} to {type(cast_value)} due to type casting."
                )
                logger.info(
                    f"Original value: {value['default']}, Casted value: {cast_value}"
                )
                if str(cast_value) != str(value["default"]):
                    logger.warning(
                        f"뭔가 사람이 보기에 값이 달라진거 같다고 생각함, PASS"
                    )
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
