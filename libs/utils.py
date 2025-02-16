def func_name_sanitizer(func_name: str) -> str:
    """
    Sanitize function name.
    """
    return func_name.replace(" ", "_").replace(".", "_")
