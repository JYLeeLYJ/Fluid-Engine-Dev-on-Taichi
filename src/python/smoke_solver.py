from Eulerian_method import GridEmitter ,Eulerian_Solver , Semi_Lagrangian , Order ,ForwardEulerDeffusionSolver,Jacobian_ProjectionSolver
from utils import DataPair , VectorField , ScalarField , clamp_index2
from basic_types import Float
import taichi as ti

@ti.data_oriented
class SmokeEmitter(GridEmitter):
    def __init__(self , temp_getter , density_getter ,source_pos , radius):
        self.get_tempreture = temp_getter
        self.get_density = density_getter
        self.source_pos = source_pos
        self.radius = radius

    @ti.func
    def smooth(self , phi :Float )-> Float:
        pi = ti.static(3.14159265358979323846264338327950288)
        invpi = ti.static(1.0 / pi)
        ret = 0.0
        if phi > 1.5 : 
            ret = 1.0
        elif phi < -1.5 :
            ret = 0.0
        else :
            ret = 0.5 + phi / 3.0 + 0.5 * invpi * ti.sin(pi * phi / 1.5)
        return ret

    @ti.kernel
    def emit(self , time_interval : Float):
        sx , sy , r= ti.static( self.source_pos[0] , self.source_pos[1] ,self.radius)
        df , tf = ti.static(self.get_density().field() , self.get_tempreture().field())
        for i , j in df :
            dx , dy = i - sx , j - sy
            d2 = dx * dx + dy * dy
            # exp = ti.exp(-d2 / r)
            # df[i,j] = max(df[i,j] , exp)
            # tf[i,j] = max(tf[i,j] , exp)
            step = 1.0 - self.smooth( d2 / r )
            df[i,j] = max(df[i,j] , step)
            tf[i,j] = max(tf[i,j] , step)

@ti.data_oriented
class FlowEmitter(GridEmitter):
    def __init__(self , density_getter , velocity_getter ,source_pos , radius):
        self.get_density = density_getter
        self.get_velocity = velocity_getter
        self.f_strength = 200.0
        self.source_x ,self.source_y = source_pos
        self.radius = radius

    @ti.kernel
    def emit(self , time_interval : Float ):
        f_strenght_dt = self.f_strength * time_interval
        inv_force_r = ti.static (1.0 / self.radius)
        # inv_dye_denom = ti.static(4.0 / (self.res / 20.0)**2)
        sx , sy = ti.static( self.source_x , self.source_y )
        vf , df= ti.static(self.get_velocity().field() , self.get_density().field())

        for i , j in vf :
            dx , dy = i + 0.5 - sx , j + 0.5 - sy
            d2 = dx * dx + dy * dy
            momentum = f_strenght_dt * ti.exp( -d2 * inv_force_r) * ti.Vector([0.0,1.0])
            vf[i,j] = vf[i,j] + momentum
            dc = df[i,j]
            dc += ti.exp(-d2 * inv_force_r ) 
            df[i,j] = min(dc , 1.0)

@ti.data_oriented
class BoxEmitter(GridEmitter):
    def __init__(self , density_getter , temp_getter , Box):
        pass

    @ti.kernel
    def emit(self , time_interval : Float):
        # for i , j in box :
        #    density = value , tempreture = value
        pass

@ti.data_oriented
class Smoke_Solver(Eulerian_Solver):
    def __init__ (self , resolution ):
        dim = len(resolution)
        # assert(dim == 2 or dim == 3 , "Incorrect resolution shape , should be (x, y) or (x,y,z).")
        # assert(dim == 2 , "only 2d resolution is supported at present , like (x,y) or [x,y]")

        self.dim = dim
        self.resolution = tuple(resolution)

        self.densityBuoyancyFactor = -0.0006
        self.tempretureBuoyancyFactor = 10.0

        # quantity field
        self._velocity_old = ti.Vector.field(2,dtype = ti.f32 , shape=self.resolution)
        self._velocity_new = ti.Vector.field(2,dtype = ti.f32 , shape=self.resolution)
        self._tempreture_old = ti.field(dtype=ti.f32 , shape=self.resolution)
        self._tempreture_new = ti.field(dtype=ti.f32 , shape=self.resolution)
        self._pressure_new = ti.field(dtype = ti.f32 , shape=self.resolution)
        self._pressure_old = ti.field(dtype = ti.f32 , shape=self.resolution)
        self._density_new = ti.field( dtype = ti.f32 , shape=self.resolution)
        self._density_old = ti.field( dtype = ti.f32 , shape=self.resolution)

        self.pvelocity = DataPair(self._velocity_old , self._velocity_new , VectorField)
        self.ptempreture = DataPair(self._tempreture_old , self._tempreture_new , ScalarField)
        self.ppressure = DataPair(self._pressure_old , self._pressure_new , ScalarField)
        self.pdensity = DataPair(self._density_old , self._density_new , ScalarField)

        super(Smoke_Solver , self).__init__(
            self.pvelocity , 
            self.ppressure ,
            Semi_Lagrangian(),
            Jacobian_ProjectionSolver(),
            ForwardEulerDeffusionSolver(0.0)
            )

        self.set_gravity([0.0 , -9.8])

    def reset(self):
        self._velocity_old.fill([0,0])
        self._velocity_new.fill([0,0])
        self._pressure_old.fill(0.0)
        self._pressure_new.fill(0.0)
        self._tempreture_old.fill(0.0)
        self._tempreture_new.fill(0.0)
        self._density_old.fill(0.0)
        self._density_new.fill(0.0)

    @ti.kernel
    def buoyancy_force(self, time_interval : ti.f32):
        n_grid = ti.static(self.resolution[0] * self.resolution[1])
        tf, vf, df = ti.static(self.tempreture().field() , self.velocity().field() , self.density().field())

        temp_avg = 0.0 
        for I in ti.grouped(tf):
            temp_avg += tf[I]
        temp_avg /= n_grid

        for I in ti.grouped(vf):
            vf[I] += time_interval * ti.Vector([0.0 , 1.0]) * \
                (self.densityBuoyancyFactor * df[I] + self.tempretureBuoyancyFactor * (tf[I] - temp_avg))

    def density(self):
        return self.pdensity.old

    def pressure(self):
        return self.ppressure.old

    def velocity(self):
        return self.pvelocity.old

    def tempreture(self):
        return self.ptempreture.old

    @ti.kernel
    def density_decay(self):
        df , tf = ti.static(self.pdensity.old.field() , self.ptempreture.old.field())
        for I in ti.grouped(df):
            df[I] = df[I] * 0.999
            tf[I] = tf[I] * 0.999

    def end_time_intergrate(self, time_interval : float):
        self.density_decay()
        super().end_time_intergrate(time_interval)

def build_smoke(resolution):
    smoke = Smoke_Solver(resolution)
    smoke.set_advection_grids([
        smoke.ptempreture,
        smoke.pdensity,
        smoke.pvelocity
    ])
    smoke.set_externel_forces([
        # smoke.gravity_force , 
        # smoke.buoyancy_force
    ])
    # smoke.set_emitter(SmokeEmitter(
    #     smoke.tempreture , 
    #     smoke.density ,
    #     (resolution[0]/2 , 0),
    #     min(resolution[0],resolution[1])))
    smoke.set_emitter(FlowEmitter(
        smoke.density , smoke.velocity ,
        (resolution[0]/2 , 0) , resolution[0]/3
    ))
    return smoke