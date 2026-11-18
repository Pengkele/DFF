from torch import nn
from torch_geometric.nn import GCNConv

from .mlp import MLP


class GCN(nn.Module):
    def __init__(
        self,
        in_feats=5,
        h_feats=64,
        num_classes=2,
        num_layers=2,
        mlp_layers=1,
        dropout_rate=0.0,
        activation="ReLU",
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.convs.append(
            GCNConv(in_feats, h_feats, add_self_loops=True, normalize=True)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(h_feats, h_feats, add_self_loops=True, normalize=True)
            )
        self.mlp = MLP(
            h_feats, h_feats, num_classes, mlp_layers, dropout_rate, activation
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr=None, graph_id=None):
        h = x
        for i, conv in enumerate(self.convs):
            if i != 0:
                h = self.dropout(h)
            h = self.act(conv(h, edge_index))
        return self.mlp(h)
