from .dff import DFF


class DFFNoTemporalConsistency(DFF):
    feature_columns = (0, 1, 2, 4)
