from .dff import DFF


class DFFNoSpatialGradient(DFF):
    feature_columns = (0, 1, 2, 3)
