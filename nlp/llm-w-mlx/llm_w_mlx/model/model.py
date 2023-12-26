from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) is a normalization technique that is similar to Layer Normalization.
    It is designed to address the problem of "mean-only" normalization techniques such as Layer Normalization and Batch Normalization.
    RMSNorm is a simple and effective normalization technique that can be used as a drop-in replacement for Layer Normalization.
    It is easy to implement and can be used in place of Layer Normalization in any application.
    https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(x.square().mean(axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Return the RMSNorm of the input x with learnable parameters.
        The input x is a tensor of shape (batch_size, sequence_length, dim).
        """
        output = self._norm(x.astype("float32")).astype(x.dtype)
        return output * self.weight


class Attention(nn.Module):
    """
    Args:
        dim: dimension of the input
        n_heads: number of heads
        n_kv_heads: number of heads for key and value
        dim_head: dimension of each head
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dim_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.repeats = n_heads // n_kv_heads

        self.scale = dim_head**-0.5

        self.wq = nn.Linear(dim, dim_head * n_heads, bias=False)
        self.wk = nn.Linear(dim, dim_head * n_kv_heads, bias=False)
        self.wv = nn.Linear(dim, dim_head * n_kv_heads, bias=False)
        self.wo = nn.Linear(dim_head * n_heads, dim, bias=False)

        self.rope = nn.RoPE(dim // n_heads, traditional=True)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> mx.array:
        """
        Args:
            x: (batch_size, sequence_length, dim)
            mask: (batch_size, sequence_length, sequence_length)
            cache: Tuple[prev_key, prev_value]
        """
        batch_size, sequences_length, dim = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.reshape(batch_size, sequences_length, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, sequences_length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, sequences_length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def _repeat(a: mx.array) -> mx.array:
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape(batch_size, self.n_heads, sequences_length, -1)

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
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(batch_size, sequences_length, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Args:
        dim: dimension of the input
        hidden_dim: dimension of the hidden layer
    """

    def __init(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Return the output of the feed forward layer.
        The input x is a tensor of shape (batch_size, sequence_length, dim).
        The output is a tensor of shape (batch_size, sequence_length, dim).
        """
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    Args:
        dim: dimension of the input
        n_heads: number of heads
        n_kv_heads: number of heads for key and value
        dim_head: dimension of each head
        hidden_dim: dimension of the hidden layer
        norm_eps: epsilon for normalization
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dim_head: int, hidden_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(dim, n_heads, n_kv_heads, dim_head)

        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward_norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[Any, Any]:
        """
        Return the output of the transformer block.
        """
        r, cache = self.attention(self.attention_norm(x), mask=mask, cache=cache)
        h = x + r
        r = self.feed_forward_norm(self.feed_forward(h))
        out = h + r
        return out, cache


class Transformer(nn.Module):
    """
    Args:
        dim: dimension of the input
        hidden_dim: dimension of the hidden layer
        vocab_size: size of the vocabulary
        n_layers: number of layers
        n_heads: number of heads
        n_kv_heads: number of heads for key and value
        head_dim: dimension of each head
        norm_eps: epsilon for normalization
    """

    def __int__(
        self,
        dim: int,
        hidden_dim: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        if self.vocab_size <= 0:
            raise ValueError("Vocab size must be greater than 0")
        self.token_embedding = nn.Embedding(self.vocab_size, dim)
        self.layers = [
            TransformerBlock(dim, n_heads, n_kv_heads, head_dim, hidden_dim, norm_eps) for _ in range(n_layers)
        ]
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, self.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Any = None) -> tuple[Any, Any]:
        """
        Return the output of the transformer.
        """
        h = self.token_embedding(inputs)
        mask = None

        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask=mask, cache=cache[e])

        return self.output(self.norm(h)), cache
