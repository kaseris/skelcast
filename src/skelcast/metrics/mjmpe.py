from skelcast.metrics import METRICS
from skelcast.metrics.metric import Metric

class MeanPerJointPositionError(Metric):
    """Mean Per Joint Position Error (MPJPE) metric.
    """
    def __init__(self, name='MPJPE', **kwargs):
        super().__init__(name=name, **kwargs)

    def update(self):
        pass

    def compute(self):
        pass