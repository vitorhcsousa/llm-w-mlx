from typing import Dict, List, Optional, Union

from .llama import LLaMAChat
from .mistral import MistralChat
from .teknium import OpenHermesChat


def create_chat(
    model_name: str,
    personality: str = "",
    examples: List[Dict[str, str]] = [],  # noqa: B006
) -> Union[MistralChat, OpenHermesChat, LLaMAChat]:
    """
    Create chat class based on model name (OpenHermes, Mistral, LLaMA) and personality.
    """

    if "openhermes" in model_name.lower():
        return OpenHermesChat(personality, examples)

    if "mistral" in model_name.lower():
        return MistralChat(personality, examples)

    if "llama" in model_name.lower():
        return LLaMAChat(personality, examples)
