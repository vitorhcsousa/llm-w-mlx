# 🚀 Large Language Models with MLX 🚀
A Python-based project that runs Large Language Models (LLM) applications on Apple Silicon in real-time thanks to [Apple MLX](https://github.com/ml-explore/mlx).

## 🌟 Features 🌟

- 🤖 Implements a Transformer model for natural language processing tasks.
- 💬 Provides a chat-like interface for interacting with the model.
- 🎭 Supports different personalities that the model can take on during a chat. You can chat with personalities like "Cristiano Ronaldo", "Joey Tribbiani", "Dwight K. Schrute", "Michael Scott", and "Chandler Bing". Each personality is defined by a description and a set of example dialogues.

## 💬 Example of a conversation with Joey Tribbiani 💬

<img src="/utils/llm_w_mlx-joey-conversation.png" width="800" height="400">

## 📥 Installation 📥

Clone the repository and install the required packages:

```bash
git clone https://github.com/vitorhcsousa/Portfolio.git
cd llm_w_mlx
pip install .
```

🚀 Usage 🚀

```bash
python llm_w_mlx/cli/llm.py --personality dwight --model Mistral-7B-Instruct-v0.1 --weights path_to_weights --tokenizer path_to_tokenizer --max_tokens 500
```





---
## 📚 Dive Deeper 📚

The LLM project revolves around the creation and interaction with a language model, specifically a Transformer model. The project is structured into several Python files, each serving a specific purpose:

- `llm.py`: This file defines the `LLM` class, which is the main class for the language model. The `LLM` class is initialized with a `Transformer` model, a `SentencePieceProcessor` tokenizer, a list of examples, a personality description, and a model name. The class has several methods for building the model, generating tokens, and interacting with the model in a chat-like interface.
- `personalities.py`: This file defines different personalities that the model can take on during a chat. Each personality is defined by a description and a set of example dialogues. Personalities include "Cristiano Ronaldo", "Joey Tribbiani", "Dwight K. Schrute", "Michael Scott", and "Chandler Bing".
- `cli/llm.py`: This file is the command-line interface for interacting with the `LLM` class. It includes argument parsing and the main function that initializes the `LLM` class and starts a chat.
- `model/llm.py`: This file contains the `LLM` class definition, which includes methods for building the model, generating tokens, and interacting with the model in a chat-like interface.
- `model/Transformer`: This class, used in the `LLM` class, represents the Transformer model used for language generation.
- `chat.py`: This file contains the `create_chat` function, which is used in the `LLM` class to create a chat instance for interacting with the model.
- `config.py`: This file contains the `CONFIG` dictionary, which includes configuration details for different models that can be used with the `LLM` class.


## 🌐 Supported Models

Here are the models supported by mlx-llm:

- **LLaMA-2**: You can download the weights for this model from [this link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). The supported model for this family is `llama-2-7b-chat`.
- **Mistral**: You can download the weights for this model from [https://docs.mistral.ai/models/](link). The supported models for this family are `Mistral-7B-Instruct-v0.1` and `Mistral-7B-Instruct-v0.2`.
- **OpenHermes-Mistral**: You can download the weights for this model from [https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/tree/main](link). The supported model for this family is `OpenHermes-2.5-Mistral-7B`.

### 🍏 Convert to Apple MLX format (.npz)

The weights of the models need to be converted to the Apple MLX format (.npz) to be used with mlx-llm. Here's how you can do it:


```python
from llm_w_mlx.utils.weights import weights_to_npz, hf_to_npz

# if weights are original ones (from raw sources)
weights_to_npz(
    ckpt_paths=[
        "path/to/model_1.bin", # also support safetensor
        "path/to/model_2.bin",
    ]
    output_path="path/to/model.npz",
)

# if weights are from HuggingFace (e.g. OpenHermes-Mistral)
hf_to_npz(
    model_name=ckpt_paths=[
        "path/to/model_1.bin", # also support safetensor
        "path/to/model_2.bin",
    ]
    output_path="path/to/model.npz",
    n_heads=32,
    n_kv_heads=8
)
```

## 🤝 Contributing 🤝
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
