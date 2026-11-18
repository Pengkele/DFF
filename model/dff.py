"""Dipolar field filter for channel classification on an IED sensor graph."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, TransformerConv


class EdgeScalarGate(nn.Module):
    def __init__(self, edge_dim=1, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, edge_dim),
            nn.Sigmoid(),
        )

    def forward(self, edge_attr):
        return edge_attr * (0.5 + self.net(edge_attr))


class ResidualFFN(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class DFF(nn.Module):
    """Predict two logits per sensor from fold-standardized features and edges."""

    conv_stack = ("gatv2", "transformer", "gatv2")
    hidden_dim = 128
    use_graphnorm = True
    # Peak value, polarity, mean neighbor amplitude, temporal consistency, gradient.
    feature_columns = (0, 1, 2, 3, 4)

    def __init__(self):
        super().__init__()
        hidden = self.hidden_dim
        dropout = 0.10
        self.edge_gate = EdgeScalarGate()
        self.input_norm = nn.LayerNorm(len(self.feature_columns))
        self.input_proj = nn.Linear(len(self.feature_columns), hidden)
        self.input_ffn = ResidualFFN(hidden, dropout)
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        kwargs = dict(heads=4, concat=False, edge_dim=1, dropout=dropout)
        for kind in self.conv_stack:
            if kind == "gatv2":
                layer = GATv2Conv(hidden, hidden, **kwargs)
            elif kind == "transformer":
                layer = TransformerConv(hidden, hidden, beta=True, **kwargs)
            else:  # MLP block in DFF-noGNNPropagation.
                layer = ResidualFFN(hidden, dropout)
            self.convs.append(layer)
        self.norms = (
            nn.ModuleList(GraphNorm(hidden) for _ in self.convs)
            if self.use_graphnorm
            else None
        )

        fusion_dim = hidden * (1 + len(self.convs))
        self.node_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, edge_index, edge_attr, graph_id):
        edge_attr = self.edge_gate(edge_attr)
        x = x[:, self.feature_columns]
        h = self.input_ffn(F.gelu(self.input_proj(self.input_norm(x))))
        states = [h]
        for i, conv in enumerate(self.convs):
            if isinstance(conv, ResidualFFN):
                h = conv(h)
            else:
                h = h + self.dropout(conv(h, edge_index, edge_attr=edge_attr))
            if self.norms is not None:
                h = self.norms[i](h, graph_id)
            h = F.gelu(h)
            states.append(h)
        return self.node_head(torch.cat(states, dim=1))
