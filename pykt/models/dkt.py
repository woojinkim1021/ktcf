"""
Script from pyKT package version v1.0.0 (https://github.com/pykt-team/pykt-toolkit).


Reference:
Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., & Luo, W. (2022). 
pyKT: a python library to benchmark deep learning based knowledge tracing models. 
Advances in Neural Information Processing Systems, 35, 18542-18555.
"""


import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

    def soft_embedding(self, embedding, indices):
        """
        Differentiable embedding lookup via linear interpolation between floor and ceil indices.
        Interpolate between their embeddings with weights proportional to how close the input is to each
            embedding(i) = emb(floor(i)) * (1 - alpha) + emb(ceil(i)) * alpha
        where alpha = i - floor(i)
        
        Args:
            embedding: nn.Embedding instance
            indices: float tensor of any shape (batch, seq)
        Returns:
            Interpolated embeddings of shape (batch, seq, emb_size)
        """
        indices = indices.clamp(0, embedding.num_embeddings - 1)
        idx_floor = torch.floor(indices)
        idx_ceil = torch.ceil(indices)
        idx_floor_long = idx_floor.long()
        idx_ceil_long = idx_ceil.long()
        emb_floor = embedding(idx_floor_long)
        emb_ceil = embedding(idx_ceil_long)
        weight_ceil = indices - idx_floor
        weight_floor = 1.0 - weight_ceil
        # Ensure weights broadcast to embedding dim
        while weight_floor.dim() < emb_floor.dim():
            weight_floor = weight_floor.unsqueeze(-1)
            weight_ceil = weight_ceil.unsqueeze(-1)
        return emb_floor * weight_floor + emb_ceil * weight_ceil

    def forward(self, q, r):
        emb_type = self.emb_type
        if emb_type == "qid":
            # q and r can be float tensors for soft embedding
            x = q + self.num_c * r  # x is float
            xemb = self.soft_embedding(self.interaction_emb, x)
        else:
            raise NotImplementedError(f"Embedding type {emb_type} not supported for soft embedding.")
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y