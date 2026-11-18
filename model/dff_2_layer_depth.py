from .dff import DFF


class DFF2LayerDepth(DFF):
    conv_stack = ("gatv2", "transformer")
