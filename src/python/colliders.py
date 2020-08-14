from abc import ABCMeta , abstractmethod
from basic_types import Vector 

from geometry import Surface
import taichi as ti

class Collider(metaclass = ABCMeta):
    @abstractmethod
    def surface(self)->Surface:
        pass

    @abstractmethod
    def update(self , time_interval: float):
        pass

    @abstractmethod
    def velocity_at(self , point : Vector) -> Vector :
        pass

# still not support self.velocity at present
# let's treat it as a static body at first
# TODO : add linear velocity and angle velocity setting , to support movement and rotation
class RigidBodyCollier(Collider):
    def __init__(self , surface : Surface):
        self._surface = surface

    def update(self , time_interval : float):
        pass

    def surface(self)->Surface:
        return self._surface

    @ti.func
    def velocity_at(self , point : Vector) -> Vector :
        #TODO : compute linear velocity and angle velocity
        return ti.Vector([0.0,0.0])