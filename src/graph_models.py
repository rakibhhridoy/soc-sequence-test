"""The hybrid architecture with message passing added on top.

The convolutional encoder, static embedding, fusion and recurrent decoder are unchanged,
so the graph layers are the only difference from the model evaluated under the fixed
protocol. Each point's representation is mixed with those of its neighbours before the
head reads it.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

import models


class GraphSOC(nn.Module):
    def __init__(self, n_dynamic: int, n_static: int, hidden: int = 32,
                 rnn_hidden: int = 32, cell: str = "lstm", dropout: float = 0.2,
                 graph_layers: int = 2, **kw):
        super().__init__()
        self.backbone = models.HybridSOC(n_dynamic=n_dynamic, n_static=n_static,
                                         hidden=hidden, rnn_hidden=rnn_hidden,
                                         cell=cell, dropout=dropout, **kw)
        self.convs = nn.ModuleList(
            [SAGEConv(rnn_hidden, rnn_hidden) for _ in range(graph_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(rnn_hidden, 2)

    def forward(self, x_dyn, x_static, edge_index):
        h = self.backbone.embed(x_dyn, x_static)
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        y = self.head(self.drop(h))
        return y[:, 0], y[:, 1]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
