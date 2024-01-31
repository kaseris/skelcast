from abc import ABC, abstractmethod

# Create abstract class Metric
class Metric(ABC):

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def compute(self):
        pass