import os
from pathlib import Path
from typing import Generator, List, Union

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from llm_w_mlx.utils.time import Timing


def smart_load(ckpt_paths: List[Union[str, Path]]) -> Generator:
    """Load a checkpoint from a list of paths.

    Args:
        ckpt_paths (List[Union[str, Path]]): list of paths to checkpoints

    Yields:
        Generator: a generator of state_dict
    """
    for ckpt_path in ckpt_paths:
        with Timing(f"> Loading checkpoint from {ckpt_path}.."):
            if ckpt_path.endswith(".safetensors"):  # type: ignore
                state_dict = {}
                with safe_open(ckpt_path, framework="pt", device="cpu") as state:  # type: ignore
                    state_dict = {k: state.get_tensor(k) for k in state.keys()}
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
        yield state_dict


def weights_to_npz(
    ckpt_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    show_kv: bool = False,
) -> None:
    """Convert a checkpoint of PyTorch or safetensors to a MLX checkpoint (npz file).

    Args:
        ckpt_path (List[Union[str, Path]]): list of paths to PyTorch checkpoint
        output_path (Union[str, Path]): path to output MLX checkpoint
        show_kv (bool, optional): show key and value shape. Defaults to False.
    """
    state_dict = {}
    for ckpt_path in ckpt_paths:
        print(f"> Loading {ckpt_path}..")
        # safetensors
        if ckpt_path.endswith(".safetensors"):  # type: ignore
            with safe_open(ckpt_path, framework="pt", device="cpu") as state:  # type: ignore
                for k in tqdm(state.keys(), total=len(state.keys()), desc="Converting.."):
                    v = state.get_tensor(k)
                    if show_kv:
                        print(k, v.shape)
                    v = v.to(torch.float16).numpy()
                    state_dict[k.replace("model.", "")] = v
        else:
            state = torch.load(ckpt_path, map_location="cpu")
            for k, v in tqdm(state.items(), total=len(state.keys()), desc="Converting.."):
                if show_kv:
                    print(k, v.shape)
                v = v.to(torch.float16).numpy()
                state_dict[k.replace("model.", "")] = v

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with Timing(f"> Saving npz file at {output_path}"):
        np.savez(output_path, **state_dict)


def hf_to_npz(
    ckpt_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    n_heads: int,
    n_kv_heads: int,
) -> None:
    """Convert a checkpoint of HuggingFace to a MLX checkpoint (npz file).

    Args:
        ckpt_paths (List[Union[str, Path]]): list of paths to HuggingFace checkpoint
        output_path (Union[str, Path]): path to output MLX checkpoint
        n_heads (int): n_heads model parameter
        n_kv_heads (int): n_kv_heads model parameter
    """

    state_dict = {}
    for state in smart_load(ckpt_paths):
        state_dict.update(state)

    layers_keys = {".".join(l.split(".")[1:3]) for l in state_dict.keys() if "layers" in l}

    keymap = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        **{f"model.{l}.input_layernorm.weight": f"{l}.attention_norm.weight" for l in layers_keys},
        **{
            f"model.{l}.self_attn.{x}_proj.weight": f"{l}.attention.w{x}.weight"
            for x in ["q", "k", "v", "o"]
            for l in layers_keys
        },
        **{f"model.{l}.post_attention_layernorm.weight": f"{l}.ffn_norm.weight" for l in layers_keys},
        **{
            f"model.{l}.mlp.{x}_proj.weight": f"{l}.feed_forward.w{y}.weight"
            for x, y in {"gate": "1", "down": "2", "up": "3"}.items()
            for l in layers_keys
        },
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    def permute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
        return w.reshape(n_heads, 2, w.shape[0] // n_heads // 2, w.shape[1]).transpose(1, 2).reshape(*w.shape[:2])

    converted_state_dict = {}
    for k, w in tqdm(state_dict.items(), total=len(state_dict), desc="Converting weights.."):
        # keep rotary_embed original eventually
        if ".rotary_embed" in k:
            continue
        if "model.layers" in k:
            if "q_proj" in k:
                w = permute(w, n_heads)
            elif "k_proj" in k:
                w = permute(w, n_kv_heads)
        converted_state_dict[keymap[k]] = w.to(torch.float16).numpy()

    del state_dict  # to save up some memory

    output_dir = os.path.dirname(output_path)
    print(f"> Saving npz file at {output_path}")
    os.makedirs(output_dir, exist_ok=True)

    with Timing(f"> Saving npz file at {output_path}"):
        np.savez(output_path, **converted_state_dict)


weights_to_npz(
    ckpt_paths=["/Volumes/VSOUSA/external-repos/llms/llama/llama-2-7b-chat/consolidated.00.pth"],
    output_path="/Volumes/VSOUSA/external-repos/llms/llama/llama-2-7b-chat/weights.npz",
)
