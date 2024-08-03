from typing import Dict, List, Optional, Union

from .llama import LLaMAChat
from .mistral import MistralChat
from .teknium import OpenHermesChat


def create_chat(
    model_name: str,
    examples: List[Dict[str, Optional[str]]],
    personality: str = "",
) -> Union[MistralChat, OpenHermesChat, LLaMAChat]:
    """
    Creates a chat instance based on the provided model name.

    Parameters:
    model_name (str): The name of the model. It can be 'openhermes', 'mistral', or 'llama'.
    examples (List[Dict[str, str]]): A list of examples to be used by the chat model.
    personality (str, optional): The personality to be used by the chat model. Defaults to an empty string.

    Returns:
    Union[MistralChat, OpenHermesChat, LLaMAChat]: An instance of the chat model based on the provided model name.

    Raises:
    ValueError: If the provided model name does not match any of the available models.
    """
    model_name = model_name.lower()
    if "openhermes" in model_name:
        return OpenHermesChat(personality, examples)

    if "mistral" in model_name:
        return MistralChat(personality, examples)

    if "llama" in model_name:
        return LLaMAChat(personality, examples)

    raise ValueError(f"Model name {model_name} not found. Available models are 'openhermes', 'mistral', and 'llama'.")
