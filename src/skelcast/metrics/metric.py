from abc import ABC, abstractmethod

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
