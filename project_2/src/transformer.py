import jax
from jax import vmap
import jax.numpy as jnp
from flax import linen as nn


class MultiHeadSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_proj = nn.Dense(self.embed_dim * 3)  # For Query, Key, Value
        self.out_proj = nn.Dense(self.embed_dim)

    def __call__(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape 
        assert embed_dim == self.embed_dim, "Input embedding dimension does not match module embedding dimension."
        
        qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_len, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)  # Each: (batch_size, seq_len, num_heads, head_dim)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)

        # Compute attention weights
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            # Apply mask: set weights to -inf where mask is 0
            attn_weights = jnp.where(mask[:, None, None, :], attn_weights, -1e9)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Compute attention output
        attn_output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

class TransformerEncoderLayer(nn.Module):
    embed_dim: int
    num_heads: int
    ff_hidden_dim: int
    #dropout_rate: float

    def setup(self):
        self.attention = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.feed_forward = nn.Sequential([
            nn.Dense(self.ff_hidden_dim),
            nn.gelu,
            nn.Dense(self.embed_dim),
        ])
        #self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, mask=None, train: bool = True):
        # Multi-head self-attention
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)# self.norm1(x + attn_out)#self.dropout(attn_out, deterministic=not train))
        # Feed-forward network
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out) #self.norm2(x + ff_out)#self.dropout(ff_out, deterministic=not train))
        return x

class Transformer(nn.Module):
    embed_dim: int
    num_heads: int
    ff_hidden_dim: int
    num_layers: int
    #dropout_rate: float

    def setup(self):
        self.layers = [
            TransformerEncoderLayer(self.embed_dim, self.num_heads, self.ff_hidden_dim)#, self.dropout_rate)
            for _ in range(self.num_layers)
        ]
        self.final_norm = nn.LayerNorm()
        self.output_layer = nn.Dense(1)  # Produces a single scalar output

    def __call__(self, x, mask, train: bool = True):
        # Apply each Transformer Encoder layer
        for layer in self.layers:
            x = layer(x, mask, train=train)
        x = self.final_norm(x)
        # Global average pooling to make the output independent of sequence length
        x = jnp.sum(x * mask[:, :, None], axis=1) #/ jnp.sum(mask, axis=1, keepdims=True)
        # Final projection to a single output
        x = jnp.sum(x, axis=1)#self.output_layer(x)
        #x = self.output_layer(x)
        return x
