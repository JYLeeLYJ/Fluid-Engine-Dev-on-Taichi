from abc import ABCMeta , abstractmethod 

class FluidSolver(metaclass = ABCMeta):
    @abstractmethod
    def advance_time_step(self , dt):
        pass

class PraticalMethod_Solver(FluidSolver):
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
    def __init__(self):
        pass

