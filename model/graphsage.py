from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
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
        self.act = getattr(nn, activation)()
        self.convs.append(SAGEConv(in_feats, h_feats, aggr=agg))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(h_feats, h_feats, aggr=agg))
        self.out = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr=None, graph_id=None):
        h = x
        for conv in self.convs:
            h = self.dropout(h)
            h = self.act(conv(h, edge_index))
        return self.out(h)
