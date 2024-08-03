import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_map, tree_unflatten
from safetensors import safe_open
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from llm_w_mlx.utils.time import Timing

from ..chat import create_chat
from .config import CONFIG
from .model import Transformer


class LLM:
    """LLM class

    Args:
        model (Transformer): a Transformer model
        tokenizer: SentencePieceProcessor tokenizer
        personality (str, optional): model personality (a description of what personality model has). Defaults to "".
        examples (List[Dict[str, Optional[str]]], optional): a list of examples of dialog [{"user": ..., "model": ...}]. Defaults to [].
        model_name (str, optional): model name. Defaults to "".
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        examples: List[Dict[str, Optional[str]]],
        personality: str = "",
        model_name: str = "",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.personality = personality
        self.examples = examples
        self.model_name = model_name

    @staticmethod
    def build(
        model_name: str,
        weights_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        examples: List[Dict[str, Optional[str]]],
        personality: str = "",
        no_rope: bool = True,
    ) -> "LLM":
        """Build an LLM model from a given model name, weights path and tokenizer path.

        Args:
            model_name (str): Mistral model name
            weights_path (Union[str, Path]): path to mlx weights
            tokenizer_path (Union[str, Path]): path to tokenizer
            personality (str, optional): Mistral personality for chat mode. Defaults to "".
            examples (List[Dict[str, Optional[str]]], optional): Mistral examples (list of {"user": ..., "model": ...} examples) for chat mode. Defaults to [].
            no_rope (bool, optional): whether to use RoPE. Defaults to True.

        Returns:
            LLM: LLM class instance with model and tokenizer
        """

        if model_name not in CONFIG.keys():
            raise ValueError(f"Model name {model_name} not found in CONFIG. Available models are {list(CONFIG.keys())}")
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights path {weights_path} does not exist.")
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer path {tokenizer_path} does not exist.")

        print(f"************ Building LLM ({model_name}) ************")

        with Timing("> Loading weights"):
            model = Transformer(**CONFIG[model_name])  # type: ignore
            weights = mx.load(weights_path)
            weights = tree_unflatten(list(weights.items()))
            weights = tree_map(lambda p: p.astype(mx.float16), weights)
            model.update(weights)

        with Timing("> Loading tokenizer"):
            tokenizer = SentencePieceProcessor(tokenizer_path)

        print("\n" + "-" * 20 + "\n")

        return LLM(model, tokenizer, examples, personality, model_name=model_name)

    def generate(self, x: mx.array, temp: Optional[float] = 0.0) -> mx.array:
        """Generate tokens from a given input

        Args:
            x (mx.array): input tokens
            temp (Optional[float], optional): model temperature. Defaults to 0.0.
        """

        def sample(logits):  # type: ignore
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        logits, cache = self.model(x[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = self.model(y[:, None], cache)
            y = sample(logits.squeeze(1))
            yield y

    def chat(self, temp: float = 0.1, max_tokens: int = 1000) -> None:
        """Chat with model

        Args:
            temp (float, optional): model temperature. Defaults to .1.
            max_tokens (int, optional): max number of tokens to generate. Defaults to 1000.
        """

        chat = create_chat(self.model_name, self.examples, self.personality)

        print("************ LLM Chat ('q' to quit, 'r' to reset) ************\n")

        while True:
            question = input("\nUser: ")
            if question == "q":
                quit()
            if question == "r":
                chat.reset()
                continue

            # adding question to dialog and getting prompt to model
            chat.add_question(question)
            prompt = chat.prompt

            x = mx.array([self.tokenizer.bos_id()] + self.tokenizer.encode(prompt))
            tokens = []
            skip = 0
            print("Model: ", end="", flush=True)
            for token in self.generate(x, temp):
                tokens.append(token)
                if len(tokens) >= max_tokens:
                    break
                mx.eval(tokens)
                token_list = [t.item() for t in tokens]
                # tokenizer sometimes fails to decode - this fixes it (it's not fancy but it works)
                try:
                    answer = self.tokenizer.decode(token_list)
                except Exception:
                    if token == self.tokenizer.vocab_size():
                        tokens[-1] = mx.array([self.tokenizer.eos_id()])
                        token_list[-1] = self.tokenizer.eos_id()
                # if answer is still prompt, continue
                status = chat.model_status(answer)
                if status == 0:
                    continue
                if status == 1:
                    skip = len(answer)
                    break
                print(answer[skip:], end="", flush=True)
                skip = len(answer)
                if token_list[-1] == self.tokenizer.eos_id():
                    break
            mx.eval(tokens)
            answer = self.tokenizer.decode([t.item() for t in tokens])
            chat.add_answer(answer)
