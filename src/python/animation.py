
from abc import ABCMeta , abstractmethod

class Animation(MetaClass = ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def pause(self):
        pass

    @abstractmethod
    def update(self):
        pass