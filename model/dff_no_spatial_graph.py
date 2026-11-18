import torch

from .dff import DFF


class DFFNoSpatialGraph(DFF):
    def forward(self, x, edge_index, edge_attr, graph_id):
        nodes = torch.arange(x.size(0), device=x.device)
        self_edges = torch.stack((nodes, nodes))
        weights = edge_attr.new_zeros((x.size(0), 1))
        return super().forward(x, self_edges, weights, graph_id)
