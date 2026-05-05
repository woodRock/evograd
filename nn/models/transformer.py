"""Transformer (Vaswani et al. 2017) — encoder-decoder seq2seq model."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.tensor import Tensor
from ..layers import Layer


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class _TransformerModule(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_enc_layers: int, num_dec_layers: int,
                 d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = _PositionalEncoding(d_model, max_seq_len, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc_layers,
                                              norm=nn.LayerNorm(d_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, d_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers,
                                              norm=nn.LayerNorm(d_model))

        self.out_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, sz: int, device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None) -> torch.Tensor:
        tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        memory  = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        out     = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out_proj(out)


class Transformer(Layer):
    """
    Full encoder-decoder Transformer for sequence-to-sequence tasks.

    Parameters
    ----------
    vocab_size    : size of the shared vocabulary
    d_model       : model dimension (embedding size)
    num_heads     : attention heads
    num_enc_layers: encoder depth
    num_dec_layers: decoder depth
    d_ff          : feed-forward hidden dimension (usually 4 × d_model)
    dropout       : dropout probability
    max_seq_len   : maximum sequence length for positional encoding

    Input:  src (B, S) and tgt (B, T) token index tensors
    Output: (B, T, vocab_size) logits Tensor
    """
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_enc_layers: int = 6, num_dec_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1,
                 max_seq_len: int = 512, device: str = "cpu"):
        self._net = _TransformerModule(
            vocab_size, d_model, num_heads,
            num_enc_layers, num_dec_layers, d_ff, dropout, max_seq_len,
        ).to(device)

    def forward(self, src: Tensor, tgt: Tensor,
                src_padding_mask: Tensor | None = None,
                tgt_padding_mask: Tensor | None = None) -> Tensor:
        skp = src_padding_mask.data if src_padding_mask is not None else None
        tkp = tgt_padding_mask.data if tgt_padding_mask is not None else None
        return Tensor(self._net(src.data, tgt.data, skp, tkp))

    def parameters(self):
        return list(self._net.parameters())

    def train(self) -> "Transformer":
        self._net.train(); return self

    def eval(self) -> "Transformer":
        self._net.eval(); return self

    @property
    def model(self) -> nn.Module:
        return self._net
