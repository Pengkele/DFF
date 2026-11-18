from .bwgnn import BWGNN
from .dff import DFF
from .dff_1_layer_depth import DFF1LayerDepth
from .dff_2_layer_depth import DFF2LayerDepth
from .dff_4_layer_depth import DFF4LayerDepth
from .dff_hidden_dim_16 import DFFHiddenDim16
from .dff_hidden_dim_256 import DFFHiddenDim256
from .dff_hidden_dim_32 import DFFHiddenDim32
from .dff_hidden_dim_64 import DFFHiddenDim64
from .dff_no_edge_attribute import DFFNoEdgeAttribute
from .dff_no_gnn_propagation import DFFNoGNNPropagation
from .dff_no_graph_norm import DFFNoGraphNorm
from .dff_no_neighbor_abs_mean import DFFNoNeighborAbsMean
from .dff_no_spatial_gradient import DFFNoSpatialGradient
from .dff_no_spatial_graph import DFFNoSpatialGraph
from .dff_no_t0_sign import DFFNoT0Sign
from .dff_no_t0_value import DFFNoT0Value
from .dff_no_temporal_consistency import DFFNoTemporalConsistency
from .dff_only_gatv2_conv import DFFOnlyGATv2Conv
from .dff_only_transformer_conv import DFFOnlyTransformerConv
from .gcn import GCN
from .gin import GIN
from .graphsage import GraphSAGE
from .mlp import MLP

MODELS = {
    "MLP": MLP,
    "GCN": GCN,
    "GIN": GIN,
    "GraphSAGE": GraphSAGE,
    "BWGNN": BWGNN,
    "DFF": DFF,
    "DFF-noGraphNorm": DFFNoGraphNorm,
    "DFF-noGNNPropagation": DFFNoGNNPropagation,
    "DFF-noEdgeAttribute": DFFNoEdgeAttribute,
    "DFF-noSpatialGraph": DFFNoSpatialGraph,
    "DFF-noT0Value": DFFNoT0Value,
    "DFF-noT0Sign": DFFNoT0Sign,
    "DFF-noNeighborAbsMean": DFFNoNeighborAbsMean,
    "DFF-noTemporalConsistency": DFFNoTemporalConsistency,
    "DFF-noSpatialGradient": DFFNoSpatialGradient,
    "DFF-onlyGATv2Conv": DFFOnlyGATv2Conv,
    "DFF-onlyTransformerConv": DFFOnlyTransformerConv,
    "DFF-1LayerDepth": DFF1LayerDepth,
    "DFF-2LayerDepth": DFF2LayerDepth,
    "DFF-4LayerDepth": DFF4LayerDepth,
    "DFF-hiddenDim16": DFFHiddenDim16,
    "DFF-hiddenDim32": DFFHiddenDim32,
    "DFF-hiddenDim64": DFFHiddenDim64,
    "DFF-hiddenDim256": DFFHiddenDim256,
}
