from .dff import DFF


class DFF4LayerDepth(DFF):
    conv_stack = ("gatv2", "transformer", "gatv2", "transformer")
