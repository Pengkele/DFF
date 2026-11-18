from torch import nn
from torch_geometric.nn import GINConv


class GIN(nn.Module):
    def __init__(
        self,
        in_feats=5,
        h_feats=64,
        num_classes=2,
        num_layers=2,
        agg="mean",
        dropout_rate=0.0,
        activation="ReLU",
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(nn.Linear(in_feats, h_feats), train_eps=True, aggr=agg)
        )
        for _ in range(1, max(num_layers - 1, 1)):
            self.convs.append(
                GINConv(nn.Linear(h_feats, h_feats), train_eps=True, aggr=agg)
            )
        self.out = GINConv(nn.Linear(h_feats, num_classes), train_eps=True, aggr=agg)
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr=None, graph_id=None):
        h = x
        for i, conv in enumerate(self.convs):
            if i != 0:
                h = self.dropout(h)
            h = self.act(conv(h, edge_index))
        h = self.dropout(h)
        return self.out(h, edge_index)
