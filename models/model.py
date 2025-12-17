from __future__ import annotations

import torch
import torch.nn as nn


class SimpleInformer(nn.Module):
    """
    A lightweight, Informer-shaped model for this repo.

    It keeps the same call signature style as Informer2020 (x_enc, x_mark_enc, x_dec, x_mark_dec),
    but implements a small Transformer encoder + linear head to predict the next `pred_len` steps.
    """

    def __init__(
        self,
        enc_in: int,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        pred_len: int = 96,
        c_out: int | None = None,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.c_out = c_out if c_out is not None else enc_in

        self.value_embedding = nn.Linear(enc_in, d_model)
        self.pos_embedding = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.proj = nn.Linear(d_model, self.c_out)

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.pred_len * self.c_out),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: [B, seq_len, enc_in]
        returns: [B, pred_len, c_out]
        """
        emb = self.value_embedding(x_enc)
        emb = self.pos_embedding(emb)
        hidden = self.encoder(emb)

        # Use the last token to predict the future window.
        last = hidden[:, -1, :]
        out = self.pred_head(last).view(x_enc.size(0), self.pred_len, self.c_out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

