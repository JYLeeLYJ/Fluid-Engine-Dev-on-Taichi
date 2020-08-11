from Eulerian_method import GridEmitter ,Eulerian_Solver , Semi_Lagrangian , Order ,Jacobian_ProjectionSolver
from utils import DataPair , VectorField , ScalarField
import taichi as ti

@ti.data_oriented
class SmokeEmitter(GridEmitter):
    def __init__(self):
        pass

    def emit(self):
        pass

@ti.data_oriented
class Smoke_Solver(Eulerian_Solver):
    def __init__ (self , resolution ):
        dim = len(resolution)
        # assert(dim == 2 or dim == 3 , "Incorrect resolution shape , should be (x, y) or (x,y,z).")
        # assert(dim == 2 , "only 2d resolution is supported at present , like (x,y) or [x,y]")

        self.dim = dim
        self.resolution = tuple(resolution)
        # self.n = self.resolution[0] * self.resolution[1] #if self.dim == 2 else \
            #self.resolution[0] * self.resolution[1] * self.resolution[2] 

        self.densityBuoyancyFactor = 1.0
        self.tempretureBuoyancyFactor = 1.0

        # quantity field
        self._velocity_old = ti.Vector.field(2,dtype = ti.f32 , shape=(self.resolution))
        self._velocity_new = ti.Vector.field(2,dtype = ti.f32 , shape=(self.resolution))
        self._tempreture_old = ti.field(dtype=ti.f32 , shape=(self.resolution))
        self._tempreture_new = ti.field(dtype=ti.f32 , shape=(self.resolution))
        self._pressure_new = ti.field(dtype = ti.f32 , shape=(self.resolution))
        self._pressure_old = ti.field(dtype = ti.f32 , shape=(self.resolution))
        self._density_new = ti.field( dtype = ti.f32 , shape=(self.resolution))
        self._density_old = ti.field( dtype = ti.f32 , shape=(self.resolution))

        self.pvelocity = DataPair(self._velocity_old , self._velocity_new , VectorField)
        self.ptempreture = DataPair(self._tempreture_old , self._tempreture_new , ScalarField)
        self.ppressure = DataPair(self._pressure_old , self._pressure_new , ScalarField)
        self.pdensity = DataPair(self._density_old , self._density_new , ScalarField)

        super(Smoke_Solver , self).__init__(
            self.pvelocity , 
            self.ppressure ,
            Semi_Lagrangian(RK = Order.RK_1),
            Jacobian_ProjectionSolver())

    def reset(self):
        self._velocity_old.fill(0.0)
        self._pressure_old.fill(0.0)
        self._tempreture_old.fill(0.0)
        self._density_old.fill(0.0)

    @ti.kernel
    def buoyancy_force(self, time_interval : ti.f32):
        n_grid = ti.static(self.resolution[0] * self.resolution[1])
        tf, vf, df = ti.static(self.ptempreture.old , self.pvelocity.old , self.pdensity.old)

        temp_avg = 0.0 
        for I in ti.grouped(tf.grid_data):
            temp_avg += tf[I]
        temp_avg /= n_grid

        for I in ti.grouped(vf.grid_data):
            vf[I] += time_interval * ( 
                - self.densityBuoyancyFactor * df.sample(I) 
                + self.tempretureBuoyancyFactor * (tf.sample(I) - temp_avg))

    def density(self):
        return self.pdensity.old

    def pressure(self):
        return self.ppressure.old

    def velocity(self):
        return self.pvelocity.old

    @ti.kernel
    def density_decay(self):
        df = ti.static(self.pdensity.old.field())
        for I in df:
            df[I] = df[I] * 0.99

    def end_time_intergrate(self, time_interval : float):
        self.density_decay()
        super().end_time_intergrate(time_interval)

def build_smoke(resolution):
    smoke = Smoke_Solver(resolution)
    smoke.set_advection_grids([
        smoke.ptempreture,
        smoke.ppressure ,
        smoke.pdensity,
        smoke.pvelocity
    ])
    smoke.set_externel_forces([
        smoke.gravity_force , 
        # smoke.buoyancy_force
    ])
    smoke.set_gravity([0.0 , -9.8])
    return smoke