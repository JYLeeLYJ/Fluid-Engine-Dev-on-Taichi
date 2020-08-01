from fluid_solver import GridMethod_Solver
from abc import ABCMeta ,abstractmethod
from utils import DataPair
from typing import List , Callable , Union , Tuple , Optional 

class AdvectionSolver(metaclass = ABCMeta):
    @abstractmethod
    def advect(self, vec_field , in_grid , out_grid , dt):
        pass

class ProjectionSolver(metatclass = ABCMeta):
    @abstractmethod
    def projection(self , vec_field , pressure_pair , dt , sdf):
        pass

class Semi_Lagrangian(AdvectionSolver):
    def __init__(self , RK = 1):
        pass

    def backtrace(self):
        pass

    def interpolate(self):
        pass

    def advect(self , vec_field , in_grid , out_grid , dt):
        pass

#TODO : bfacc
class BFACC(AdvectionSolver):
    def  __init__(self):
        pass

    def advect(self, vec_field , in_grid , out_grid , dt):
        pass

class Eulerian_Solver(GridMethod_Solver):
    def __init__(
        self ,
        advection_solver    : AdvectionSolver ,
        projection_solver   : ProjectionSolver ,
        velocity_pair       : DataPair,
        pressure_pair       : DataPair,
        ):
        
        self.advection_solver = advection_solver 
        self.projection_solver= projection_solver

        self._velocity_pair = velocity_pair
        self._pressure_pair = pressure_pair


    def set_advection_grids(self , grids: List[DataPair]):
        assert(isinstance(grids , list) , "type of grids should be List[DataPair].")
        self.advection_grids = grids

    def set_externel_forces(self , forces : List[Callable[[float] , None]]) :
        assert(isinstance(forces , list) , "type of forces should be List[Callable[[float] , None]]")
        self.force_calculators = forces

    def compute_advection(self, time_interval : float):
        for pair in self.advection_grids :
            self.advection_solver.advect(self._velocity_pair.old , pair , time_interval)
        for pair in self.advection_grids :
            pair.swap()
        #TODO: apply boundary condition
        pass

    def compute_external_force(self , time_interval : float):
        for f in self.force_calculators:
            f(time_interval)    # update velocity field

    def compute_projection(self , time_interval : float):
        #TODO
        pass

    def compute_viscosity(self):
        pass
