import math

import torch
from torch import nn

from .mlp import MLP


def beta_fn(a, b):
    return math.gamma(a) * math.gamma(b) / math.gamma(a + b)


def poly_mul(a, b):
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def calculate_theta(d):
    thetas = []
    for i in range(d + 1):
        left = [0.0] * i + [(0.5) ** i]
        right = [math.comb(d - i, j) * ((-0.5) ** j) for j in range(d - i + 1)]
        coeff = poly_mul(left, right)
        coeff = [c / beta_fn(i + 1, d + 1 - i) for c in coeff]
        coeff = coeff + [0.0] * (d + 1 - len(coeff))
        thetas.append(coeff[: d + 1])
    return thetas


class PolyConvPyG(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.register_buffer("theta", torch.tensor(theta, dtype=torch.float32))

    def normalized_laplacian(self, feat, edge_index):
        row, col = edge_index
        n = feat.shape[0]
        deg = torch.zeros(n, dtype=feat.dtype, device=feat.device)
        deg.index_add_(
            0, col, torch.ones(col.numel(), dtype=feat.dtype, device=feat.device)
        )
        deg = deg.clamp_min(1.0)
        norm = deg[row].pow(-0.5) * deg[col].pow(-0.5)
        agg = torch.zeros_like(feat)
        agg.index_add_(0, col, feat[row] * norm.unsqueeze(-1))
        return feat - agg

    def forward(self, x, edge_index):
        theta = self.theta.to(dtype=x.dtype, device=x.device)
        h = theta[0] * x
        feat = x
        for k in range(1, int(theta.numel())):
            feat = self.normalized_laplacian(feat, edge_index)
            h = h + theta[k] * feat
        return h


class BWGNN(nn.Module):
    def __init__(
        self,
        in_feats=5,
        h_feats=64,
        num_classes=2,
        num_layers=2,
        mlp_layers=2,
        dropout_rate=0.0,
        activation="ReLU",
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.act = getattr(nn, activation)()
        self.convs = nn.ModuleList(
            [PolyConvPyG(theta) for theta in calculate_theta(num_layers)]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp = MLP(
            h_feats * len(self.convs),
            h_feats,
            num_classes,
            mlp_layers,
            dropout_rate,
            activation,
        )

    def forward(self, x, edge_index, edge_attr=None, graph_id=None):
        h = self.act(self.linear1(x))
        h = self.act(self.linear2(h))
        h_final = torch.cat([conv(h, edge_index) for conv in self.convs], dim=-1)
        h_final = self.dropout(h_final)
        return self.mlp(h_final)
