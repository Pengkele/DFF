from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_feats=5,
        h_feats=64,
        num_classes=2,
        num_layers=2,
        dropout_rate=0.0,
        activation="ReLU",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers <= 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for _ in range(1, num_layers - 1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index=None, edge_attr=None, graph_id=None):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers) - 1:
                h = self.act(h)
        return h
