from .dff import DFF


class DFFOnlyTransformerConv(DFF):
    conv_stack = ("transformer",) * 3
