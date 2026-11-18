from .dff import DFF


class DFFNoGNNPropagation(DFF):
    conv_stack = ("mlp",) * 3
