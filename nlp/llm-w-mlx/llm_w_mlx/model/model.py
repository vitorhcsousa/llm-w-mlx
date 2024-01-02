from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

__all__ = ["Transformer"]


# Root Mean Square Layer Normalization (RMSNorm).
class RMSNorm(nn.Module):
    """
    This class implements the RMSNorm layer. RMSNorm is a type of layer normalization
    that normalizes the input tensor along the last dimension.

    Args:
        dims (int): The number of dimensions in the input tensor.
        eps (float, optional): A small constant for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x: mx.array) -> mx.array:
        """
        Normalizes the input tensor.

        Args:
            x (mx.array): The input tensor.

        Returns:
            mx.array: The normalized tensor.
        """
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Applies the RMSNorm to the input tensor.

        Args:
            x (mx.array): The input tensor.

        Returns:
            mx.array: The output tensor after applying RMSNorm.
        """
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


# Multi-head attention with RoPE and RMSNorm.
class Attention(nn.Module):
    """
    This class implements the Attention mechanism with RoPE (Rotary Positional Encoding)
    and RMSNorm. It is a type of multi-head attention mechanism where the input tensor
    is transformed into queries, keys, and values before the attention scores are computed.

    Args:
        dim (int): The number of dimensions in the input tensor.
        n_heads (int): The number of attention heads.
        n_kv_heads (int): The number of key/value heads.
        head_dim (int): The dimensionality of each head.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()

        self.n_heads: int = n_heads
        self.n_kv_heads: int = n_kv_heads

        self.repeats = n_heads // n_kv_heads

        self.scale = head_dim**-0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(dim // n_heads, traditional=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Applies the Attention mechanism to the input tensor.

        Args:
            x (mx.array): The input tensor.
            mask (Optional[mx.array], optional): The attention mask. Defaults to None.
            cache (Optional[Tuple[mx.array, mx.array]], optional): The cache for fast inference. Defaults to None.

        Returns:
            mx.array: The output tensor after applying the Attention mechanism.
        """
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a: mx.array) -> mx.array:
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


# Feed forward network with SiLU activation.
class FeedForward(nn.Module):
    """
    This class implements a Feed Forward Network (FFN) with SiLU (Sigmoid Linear Unit) activation.
    It is a type of neural network layer where the input is fully connected to the output.

    Args:
        dim (int): The number of dimensions in the input tensor.
        hidden_dim (int): The number of dimensions in the hidden layer.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        """
        Applies the Feed Forward Network to the input tensor.

        Args:
            x (mx.array): The input tensor.

        Returns:
            mx.array: The output tensor after applying the Feed Forward Network.
        """
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# Transformer block with RMSNorm and RoPE.
class TransformerBlock(nn.Module):
    """
    This class implements a Transformer block with RMSNorm and RoPE.
    It is a type of neural network layer that combines a multi-head attention mechanism
    with a feed forward network.

    Args:
        dim (int): The number of dimensions in the input tensor.
        n_heads (int): The number of attention heads.
        n_kv_heads (int): The number of key/value heads.
        head_dim (int): The dimensionality of each head.
        hidden_dim (int): The number of dimensions in the hidden layer.
        norm_eps (float): A small constant for numerical stability in the RMSNorm layer.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int, norm_eps: float):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Applies the Transformer block to the input tensor.

        Args:
            x (mx.array): The input tensor.
            mask (Optional[mx.array], optional): The attention mask. Defaults to None.
            cache (Optional[Tuple[mx.array, mx.array]], optional): The cache for fast inference. Defaults to None.

        Returns:
            mx.array: The output tensor after applying the Transformer block.
        """
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


# Transformer model with RMSNorm and RoPE.
class Transformer(nn.Module):
    """
    This class implements a Transformer model with RMSNorm and RoPE.
    It is a type of neural network that uses self-attention mechanisms and is designed
    to handle sequential data.

    Args:
        dim (int): The number of dimensions in the input tensor.
        hidden_dim (int): The number of dimensions in the hidden layer.
        vocab_size (int): The size of the vocabulary.
        n_layers (int): The number of layers in the Transformer.
        n_heads (int): The number of attention heads.
        n_kv_heads (int): The number of key/value heads.
        head_dim (int): The dimensionality of each head.
        norm_eps (float): A small constant for numerical stability in the RMSNorm layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = [
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Applies the Transformer model to the input tokens.

        Args:
            inputs (mx.array): The input tokens.
            cache (Optional[Tuple[mx.array, mx.array]], optional): The cache for fast inference. Defaults to None.

        Returns:
            Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]: The output tokens and the cache.
        """
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h)), cache
