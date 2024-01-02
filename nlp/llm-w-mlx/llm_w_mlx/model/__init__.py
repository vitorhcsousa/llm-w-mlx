from typing import List

from .llm import LLM


def list_models() -> List[str]:
    from .config import CONFIG

    return list(CONFIG.keys())
