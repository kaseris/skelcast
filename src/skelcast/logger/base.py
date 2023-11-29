
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    @abstractmethod
    def add_scalar(self):
        pass

    @abstractmethod
    def add_scalars(self):
        pass
    
    @abstractmethod
    def add_histogram(self):
        pass

    @abstractmethod
    def add_image(self):
        pass

    @abstractmethod
    def close(self):
        pass
