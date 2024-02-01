from abc import ABC, abstractmethod
from skelcast.metrics import METRICS


class Metric(ABC):
    @abstractmethod
    def update(self, predictions, targets):
        """
        Update the metric's state using the predictions and the targets.

        Args:
        -    predictions: The predicted values.
        -    targets: The ground truth values.
        """
        pass

    @abstractmethod
    def result(self):
        """
        Calculates and returns the final metric result based on the state.

        Returns:
        -    The calculated metric.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the metric state.
        """
        pass


def create_metric(metric_name, **kwargs):
    metric_class = METRICS.get_module(metric_name)
    return metric_class(**kwargs)


class MetricList:
    def __init__(self, metric_names):
        self.metrics = [create_metric(name) for name in metric_names]

    def update(self, output):
        for metric in self.metrics:
            metric.update(output)

    def compute(self):
        return {type(metric).__name__: metric.result() for metric in self.metrics}

    def reset(self):
        for metric in self.metrics:
            metric.reset()
