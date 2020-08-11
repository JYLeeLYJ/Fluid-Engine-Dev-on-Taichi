
from abc import ABCMeta , abstractmethod

class Animation(metaclass = ABCMeta):
    @abstractmethod
    def display(self):
        pass

    # @abstractmethod
    # def pause(self):
    #     pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self):
        pass