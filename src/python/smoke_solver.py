from Eulerian_method import GridEmitter ,Eulerian_Solver , Semi_Lagrangian , Order ,ForwardEulerDeffusionSolver,Jacobian_ProjectionSolver
from utility import DataPair , VectorField , ScalarField 
from helper_functions import clamp_index2
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
            # df[i,j] = max(df[i,j] , step)
            # tf[i,j] = max(tf[i,j] , step)
            df[i,j] = min(df[i,j] + step , 1.0)
            tf[i,j] = min(tf[i,j] + step , 1.0)

@ti.data_oriented
class FlowEmitter(GridEmitter):
    def __init__(self , smoke ,source_pos , radius , f_strength):
        self.get_density = smoke.density
        self.get_velocity = smoke.velocity
        self.f_strength = f_strength
        self.source_x ,self.source_y = source_pos
        self.radius = radius

    @ti.kernel
    def emit(self , time_interval : Float ):
        f_strenght_dt = self.f_strength * time_interval
        inv_force_r = ti.static (1.0 / self.radius)
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

"""
@ti.data_oriented
class VolumnEmitter(GridEmitter):
    def __init__(self , density_getter , temp_getter , Box):
        pass

    @ti.kernel
    def emit(self , time_interval : Float):
        # for i , j in box :
        #    density = value , tempreture = value
        pass
"""

@ti.data_oriented
class Smoke_Solver(Eulerian_Solver):
    def __init__ (self , resolution):
        dim = len(resolution)

        self.dim = dim
        self.resolution = tuple(resolution)

        # quantity field
        self._velocity_old = ti.Vector.field(2,dtype = ti.f32 , shape=self.resolution)
        self._velocity_new = ti.Vector.field(2,dtype = ti.f32 , shape=self.resolution)
        self._tempreture_old = ti.field(dtype=ti.f32 , shape=self.resolution)
        self._tempreture_new = ti.field(dtype=ti.f32 , shape=self.resolution)
        self._pressure_new = ti.field(dtype = ti.f32 , shape=self.resolution)
        self._pressure_old = ti.field(dtype = ti.f32 , shape=self.resolution)
        self._density_new = ti.field( dtype = ti.f32 , shape=self.resolution)
        self._density_old = ti.field( dtype = ti.f32 , shape=self.resolution)

        self.pvelocity = DataPair(self._velocity_old , self._velocity_new , VectorField , self.resolution)
        self.ptempreture = DataPair(self._tempreture_old , self._tempreture_new , ScalarField , self.resolution)
        self.ppressure = DataPair(self._pressure_old , self._pressure_new , ScalarField,self.resolution)
        self.pdensity = DataPair(self._density_old , self._density_new , ScalarField,self.resolution)

        super().__init__(
            self.resolution,
            self.pvelocity , 
            self.ppressure ,
            Semi_Lagrangian(Order.RK_3),
            Jacobian_ProjectionSolver(),
            ForwardEulerDeffusionSolver(0.0)
            )

        self.set_advection_grids([
            self.ptempreture,
            self.pdensity,
            self.pvelocity
        ])

        self.set_gravity([0.0 , -9.8])

        self._decay = 1.0

    @property
    def density_factor(self):
        return  self._density_buoyancy_factor 

    @density_factor.setter
    def density_factor(self , factor ):
        self._density_buoyancy_factor  = factor

    @property
    def tempreture_factor(self ):
        return self._tempreture_buoyancy_factor

    @tempreture_factor.setter
    def tempreture_factor(self , factor ):
        self._tempreture_buoyancy_factor = factor

    @property
    def decay(self ):
        return self._decay

    @decay.setter
    def decay(self , decay_factor):
        self._decay = decay_factor

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
                (self.density_factor * df[I] + self.tempreture_factor * (tf[I] - temp_avg))

    def density(self):
        return self.pdensity.old

    def tempreture(self):
        return self.ptempreture.old

    @ti.kernel
    def apply_decay(self):
        df , tf = ti.static(self.pdensity.old.field() , self.ptempreture.old.field())
        for I in ti.grouped(df):
            df[I] = df[I] * self.decay
            tf[I] = tf[I] * self.decay

    def end_time_intergrate(self, time_interval : float):
        self.apply_decay()
        super().end_time_intergrate(time_interval)


class Smoke_Builder:
    def __init__(self , resolution):
        self._smoke = Smoke_Solver(resolution) 
        self._forces = []

    def build(self) :
        self._smoke.set_externel_forces(self._forces)
        return self._smoke

    def set_compute_buoyancy_force(self ,density_factor = -0.0006 , tempreture_factor= 300.0):
        self._smoke.density_factor = density_factor
        self._smoke.tempreture_factor = tempreture_factor
        self._forces.append(self._smoke.buoyancy_force)

        return self
    
    def set_compute_gravity(self) :
        self._forces.append(self._smoke.gravity_force)
        return self

    def set_decay(self ,decay = 0.99 ):
        self._smoke.decay = decay
        return self

    def add_flow_emitter(self,source_pos , radius , f_strenght = 1000.0) :
        self._smoke.add_emitter(FlowEmitter(self._smoke , source_pos ,radius , f_strenght))
        return self


