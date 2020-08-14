from abc import ABCMeta , abstractmethod
from basic_types import Vector 

from geometry import Surface , SurfaceToImplict , ImplicitSurface
import taichi as ti

class Collider(metaclass = ABCMeta):
    @abstractmethod
    def surface(self)->Surface:
        pass

    @abstractmethod
    def implict_surface(self)->ImplicitSurface:
        pass

    @abstractmethod
    def update(self , time_interval: float):
        pass

    @abstractmethod
    def velocity_at(self , point : Vector) -> Vector :
        pass

# still not support self.velocity at present
# let's treat it as a static body at first
# TODO : add linear velocity and angle velocity setting , to support translation and rotation motion
@ti.data_oriented
class RigidBodyCollier(Collider):
    def __init__(self , surface : Surface):
        self._surface = surface
        self._implicit = SurfaceToImplict(surface) if not isinstance(surface , ImplicitSurface) else surface

    def update(self , time_interval : float):
        pass    #do nothing

    def surface(self)->Surface:
        return self._surface

    def implict_surface(self)->ImplicitSurface:
        return self._implicit

    @ti.func
    def velocity_at(self , point : Vector) -> Vector :
        #TODO : compute linear velocity and angle velocity
        return ti.Vector([0.0,0.0])