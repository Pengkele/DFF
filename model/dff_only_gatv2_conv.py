from .dff import DFF


class DFFOnlyGATv2Conv(DFF):
    conv_stack = ("gatv2",) * 3
