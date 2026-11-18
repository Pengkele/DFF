import torch

from .dff import DFF


class DFFNoEdgeAttribute(DFF):
    def forward(self, x, edge_index, edge_attr, graph_id):
        # A unit edge attribute becomes zero after fold standardization.
        return super().forward(x, edge_index, torch.zeros_like(edge_attr), graph_id)
