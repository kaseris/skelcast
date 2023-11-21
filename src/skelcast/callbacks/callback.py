import abc


class Callback(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractclassmethod
    def on_batch_start(self):
        pass
    
    @abc.abstractclassmethod
    def on_batch_end(self):
        pass
    
    @abc.abstractclassmethod
    def on_epoch_start(self):
        pass

    @abc.abstractclassmethod
    def on_epoch_end(self):
        pass
    