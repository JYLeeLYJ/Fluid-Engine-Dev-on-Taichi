from Eulerian_method import Eulerian_Solver
from utils import DataPair , GridSampler , DirectlySampler 
import taichi as ti

class Smoke_Solver(Eulerian_Solver):
    def __init__ (self , resolution ):
        dim = len(resolution)
        #assert(dim == 2 or dim == 3 , "Incorrect resolution shape , should be (x, y) or (x,y,z).")
        assert(dim == 2 , "only 2d resolution is supported at present , like (x,y) or [x,y]")

        self.dim = dim
        self.resolution = tuple(resolution)
        self.n = self.resolution[0] * self.resolution[1] #if self.dim == 2 else \
            #self.resolution[0] * self.resolution[1] * self.resolution[2] 

        self.densityBuoyancyFactor = 1.0
        self.tempretureBuoyancyFactor = 1.0

        # quantity field
        self._velocity_old = ti.field(dtype = ti.f32 , shape=(self.resolution))
        self._velocity_new = ti.field(dtype = ti.f32 , shape=(self.resolution))
        self._tempreture_old = ti.field(dtype=ti.f32 , shape=(self.resolution))
        self._tempreture_new = ti.field(dtype=ti.f32 , shape=(self.resolution))
        self._pressure_new = ti.field(dtype = ti.f32 , shape=(self.resolution))
        self._pressure_old = ti.field(dtype = ti.f32 , shape=(self.resolution))
        self._density_new = ti.field( dtype = ti.f32 , shape=(self.resolution))
        self._density_old = ti.field( dtype = ti.f32 , shape=( self.resolution))

        # data pairs
        sampler = DirectlySampler

        self.velocity = DataPair(self._velocity_old , self._velocity_new , sampler)
        self.tempreture = DataPair(self._tempreture_old , self._tempreture_new , sampler)
        self.pressure = DataPair(self._pressure_old , self._pressure_new , sampler)
        self.density = DataPair(self._density_old , self._density_new , sampler)

        pass

    def set_gravity(self , g ):
        self.gravity = g

    @ti.kernel
    def compute_gravity(self , time_interval : ti.f32):
        vf = ti.static(self.velocity.old)
        for I in ti.grouped(vf.grid_data):
            vf[I] += self.gravity * time_interval

    @ti.kernel
    def compute_buoyancy(self, time_interval : ti.f32):
        n_grid = ti.static(self.n)
        tf, vf, df = ti.static(self.tempreture.old , self.velocity.old , self.density.old)

        temp_avg = 0.0 
        for I in ti.grouped(tf.grid_data):
            temp_avg += tf[I]
        temp_avg /= n_grid

        for I in ti.grouped(vf.grid_data):
            vf[I] += time_interval * ( 
                - self.densityBuoyancyFactor * df.sample(I) 
                + self.tempretureBuoyancyFactor * (tf.sample(I) - temp_avg))


#TODO
class SmokeSolver_Builder:
    def __init__(self):
        pass

