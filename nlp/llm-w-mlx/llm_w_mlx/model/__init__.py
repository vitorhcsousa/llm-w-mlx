from .llm import LLM


def list_models() -> list:
    from .config import CONFIG

    return list(CONFIG.keys())
