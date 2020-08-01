from fluid_solver import GridMethod_Solver
from abc import ABCMeta ,abstractmethod
from utils import DataPair

class Grid:
    @property
    def resolution(self):
        pass

    @property
    def origin(self):
        pass

    @property
    def grid_spacing(self):
        pass

    @property
    def bounding_box(self):
        pass

class GridSampler(metaclass = ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self , grid):
        pass

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
        gravity             
        ):
        
        self.advection_solver = advection_solver 
        self.projection_solver= projection_solver

        self.gravity = gravity
        self.velocity_pair = velocity_pair
        self.pressure_pair = pressure_pair

    def compute_advection(self, time_interval):
        self.advection_solver.advect(self.velocity_pair.old , self.velocity_pair , time_interval)
        self.velocity_pair.swap()
        #TODO: apply boundary condition
        pass

    def compute_external_force(self , time_interval):
        #TODO: 
        # 1. compute gravity
        # 2. other momentum
        pass

    def compute_projection(self , time_interval):
        #TODO
        pass

    def compute_viscosity(self):
        pass
