from abc import ABCMeta , abstractmethod 
from enum import Enum

class FluidSolver(metaclass = ABCMeta):
    @abstractmethod
    def advance_time_step(self , dt):
        pass

class PraticalMethod_Solver(FluidSolver):
    #TODO
    def __init__(self):
        pass

class GridMethod_Solver(FluidSolver):
    @abstractmethod
    def begin_time_intergrate(self):
        pass

    @abstractmethod
    def compute_external_force(self , time_interval):
        pass

    @abstractmethod
    def compute_viscosity(self):
        pass

    @abstractmethod
    def compute_projection(self , time_interval):
        pass

    @abstractmethod
    def compute_advection(self , time_interval):
        pass

    @abstractmethod
    def end_time_intergrate(self):
        pass

    def advance_time_step(self , time_interval):
        self.begin_time_intergrate()
        self.compute_advection(time_interval)
        self.compute_external_force(time_interval)
        self.compute_viscosity()
        self.compute_projection(time_interval)
        self.end_time_intergrate()

class HybridMethod_Solver(FluidSolver):
    #TODO
    def __init__(self):
        pass

### =================================================================================

class FluidMark(Enum):
    Fluid = 0
    Air = 1
    Boundary = 2

class AdvectionSolver(metaclass = ABCMeta):
    @abstractmethod
    def advect(self, vec_field , in_grid , out_grid , dt):
        pass

class ProjectionSolver(metatclass = ABCMeta):
    @abstractmethod
    def projection(self , vec_field , pressure_pair , dt , sdf):
        pass

class DiffusionSolver(metaclass = ABCMeta):
    @abstractmethod
    def solve(self , vec_field , next_vec_field , diffusionCoefficient , dt) :
        pass
