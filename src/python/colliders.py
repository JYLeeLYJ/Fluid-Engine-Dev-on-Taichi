from abc import ABCMeta , abstractmethod

class Collider(metaclass = ABCMeta):
    @abstractmethod
    def surface(self):
        pass

    @abstractmethod
    def update(self , time_interval: float):
        pass
