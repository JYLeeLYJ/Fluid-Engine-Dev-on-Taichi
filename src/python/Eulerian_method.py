from abc import ABCMeta ,abstractmethod
from typing import List , Callable , Union , Tuple , Optional 
from enum import Enum

from utility import *
from helper_functions import *
from fluid_solver import FluidMark ,GridMethod_Solver , AdvectionSolver , ProjectionSolver , DiffusionSolver , GridBoudaryConditionSolver
from basic_types import Float , Vector ,Index
from colliders import Collider
from geometry import Surface , SurfaceToImplict , ImplicitSurface

import taichi as ti

class GridEmitter(metaclass = ABCMeta):
    @abstractmethod
    def emit(self):
        pass

class Order(Enum):
    RK_1 = 1
    RK_2 = 2
    RK_3 = 3

@ti.data_oriented
class Semi_Lagrangian(AdvectionSolver):

    def __init__(self , RK = Order.RK_1):
        assert isinstance(RK ,Order)
        self.RK = RK

    @ti.func
    def backtrace(
        self ,  
        vel_grid : ti.template() ,
        boundary_sdf : ScalarField ,
        I : Index, 
        dt : Float ) -> Vector:
        
        beg = ti.Vector([float(I[0]) + 0.5 , float(I[1]) + 0.5])
        p = beg
        if ti.static(self.RK == Order.RK_1):
            p -= dt * vel_grid.value(I)
        elif ti.static(self.RK == Order.RK_2):
            mid = p - 0.5 * dt * vel_grid.value(I)
            p -= dt * vel_grid.sample(mid)
        elif ti.static(self.RK == Order.RK_3):
            v1 = vel_grid.value(I)
            p1 = p - 0.5 * dt * v1
            v2 = vel_grid.sample(p1)
            p2 = p - 0.75 * dt * v1
            v3 = vel_grid.sample(p2)
            p -= dt * ( 2/9 * v1 + 1/3 * v2 + 4/9 * v3)

        # boundary handling
        phi_beg = boundary_sdf.sample(beg)
        phi_end = boundary_sdf.sample(p)

        if phi_beg * phi_beg < 0.0 :    # inside sdf
            w = ti.abs(phi_end) / (ti.abs(phi_beg) + ti.abs(phi_end))
            p = linear_interpolate(beg , p , w)

        return p

    @ti.kernel
    def advect(
        self , 
        vel_grid : ti.template(),   # Grid (VectorField)
        in_grid : ti.template() ,   # Grid
        out_grid : ti.template() ,  # Grid
        sdf : ti.template() ,       # ScalarField
        dt : Float):

        out = ti.static(out_grid.field())

        for I in ti.grouped(vel_grid.field()):
            pos = self.backtrace(vel_grid ,sdf, I ,dt)
            out[I] = in_grid.sample(pos)

@ti.data_oriented
class ForwardEulerDeffusionSolver(DiffusionSolver):
    def __init__(self , diffusion_coefficient):
        self.coefficient = diffusion_coefficient
    
    @ti.kernel
    def solve(
        self , 
        vel_grid : ti.template() ,      # VectorField
        next_vel_grid : ti.template() , # VectorField
        marker : ti.template(),         # ScalaField
        dt : Float ):

        vf_cur , vf_nxt = ti.static(vel_grid.field() , next_vel_grid.field())
        for I in ti.grouped(vf_cur) :
            if marker.value(I) == FluidMark.Fluid:
                vf_nxt[I] = vf_cur[I] + self.coefficient * dt * self.laplacian(vel_grid ,marker, I)
            else :
                vf_nxt[I] = vf_cur[I]

    # TODO : reduce to gird.laplacian()
    @ti.func
    def laplacian( self , grid : Grid, marker : Grid , I : Index) :
        sz , ds = grid.size() , grid.spacing()
        dfx , dfy = ti.Vector([0.0 , 0.0]) , ti.Vector([0.0 , 0.0])
        i , j = I[0] , I[1]
        center = grid.value(I)
        if i > 0 and marker.value([i - 1,j]) == FluidMark.Fluid :
            dfx += grid.value([i - 1, j]) - center
        elif i + 1 < sz[0] and marker.value(ti.Vector([i+1 , j])) == FluidMark.Fluid :
            dfx += grid.value([i + 1, j]) - center

        if j > 0 and marker.value([ i , j -1 ]) == FluidMark.Fluid:
            dfy += grid.value([i , j - 1]) - center
        elif j + 1 < sz[1] and marker.value([i , j+1]) == FluidMark.Fluid :
            dfy += grid.value([i , j + 1]) - center
        
        return dfx / (ds[0] ** 2) + dfy / (ds[1] ** 2) 

@ti.data_oriented
class Jacobian_ProjectionSolver(ProjectionSolver):
    def __init__(self , max_iter : int = 30):
        self._max_iter = max_iter

    def projection(
        self ,
        vel_field : VectorField,
        pressure_pair : DataPair,
        marker : ScalarField,
        dt : Float ):

        for _ in range(self._max_iter) :
            self.jacobian_step( vel_field ,pressure_pair.old , pressure_pair.new , dt) 
            pressure_pair.swap()

        self.update_v(vel_field , pressure_pair.old  , marker,dt)

    @ti.kernel
    def update_v(
        self , 
        vel_field : ti.template() ,     # VectorField
        pressure_field : ti.template() ,# ScalarField
        marker : ti.template(),         # ScalarField
        time_interval : Float ):

        # TODO : use marker in divergence
        v = ti.static(vel_field.field())
        for I in ti.grouped(v):
            # treat (delta_t / density) is constant while doing projection 
            v[I] -= pressure_field.gradient(I) # time_interval / density.value(I)

    @ti.kernel
    def jacobian_step(        
        self,
        vel_field : ti.template() ,
        pressure_curr   : ti.template() , 
        pressure_next   : ti.template() ,
        dt : Float ):

        pf_nxt = ti.static(pressure_next.field())
        sz = ti.static(vel_field.size())
        for i , j in vel_field.field() :
            pl = pressure_curr.value(clamp_index2(i - 1 , j , sz))    
            pr = pressure_curr.value(clamp_index2(i + 1 , j , sz))
            pt = pressure_curr.value(clamp_index2(i , j + 1 , sz))
            pb = pressure_curr.value(clamp_index2(i , j - 1 , sz))
            pf_nxt[i,j] = 0.25 *  (pl + pr + pt + pb - vel_field.divergence([i,j])) 

@ti.data_oriented
class STDBoundarySolver(GridBoudaryConditionSolver):
    def __init__(
        self ,
        size : Tuple[int] , 
        is_open : bool = False , 
        extrapolation_depth : int= 5):

        self.size = size
        self.boundary_open = is_open
        self.extrapolation_depth = 5

        self._collider_sdf = ScalarField(ti.field(dtype = ti.f32 , shape = size),size)
        self._collider_vel = VectorField(ti.Vector.field(2, dtype= ti.f32 , shape = size),size)

    def constrain_velocity(self , pvelocity : DataPair , marker :DataPair ,depth : int ):
        
        # 1.build markers
        self.build_marker(marker.old.field() , pvelocity.old.field())
        extrapolate_to_region(pvelocity.old , pvelocity.new , marker , self.extrapolation_depth)
        pvelocity.swap()
        
        # 2.No flux and Slip
        self.process_flux_and_slip(pvelocity.old.field() , pvelocity.new.field())
        pvelocity.swap()
        
        # 3.check OpenBoundary => set velocity to be zero
        if self.boundary_open is False :
            self.close_boundary(pvelocity.old)

    def update_collider(self , colliders : List[Collider] ):
        self._collider_sdf.field().fill(1e8)    # fill max value
        for cld in colliders :
            surface = cld.implict_surface()
            self.update_fields(cld , surface) 

    @ti.kernel
    def build_marker(self ,marker : ti.template() , velocity : ti.template()):
        for I in ti.grouped(velocity):
            if insideSdf(self.collider_sdf().value(I)) :
                velocity[I] = self.collider_velocity_at(I)
                marker[I] = 0
            else :
                marker[I] = 1

    @ti.kernel
    def process_flux_and_slip(self , in_vel : ti.template() , out_vel : ti.template()):
        for I in ti.grouped(in_vel):
            if insideSdf(self.collider_sdf().value(I)) :
                collider_vel = self.collider_velocity_at(I)
                vel = in_vel[I]
                grad = self.collider_sdf().gradient(I)
                if grad.norm() > 0 :
                    normal = grad.normalized()
                    velr = vel - collider_vel   # relative velocity 
                    out_vel[I] = self.apply_constrain(velr , normal) + collider_vel
                else :
                    out_vel[I] = collider_vel
            else:
                out_vel[I] = in_vel[I]

    @ti.kernel
    def close_boundary(self , velocity : ti.template()):
        vf = ti.static(velocity.field())
        sz = ti.static(velocity.size())
        for i in range(sz[0]):
            vf[i,0][1] = 0.0
            vf[i,sz[1]-1][1] = 0.0

        for j in range(sz[1]):
            vf[0,j][0] = 0.0
            vf[sz[0] - 1, j][0] = 0.0

    @ti.func
    def apply_constrain(self , vel : Vector, normal : Vector) -> Vector:
        # just no-flux condition here 
        return vel - (vel.dot(normal)) * normal

    @ti.func
    def collider_velocity_at(self , I : Index) -> Vector:
        return self._collider_vel.value(I)

    def collider_sdf(self) -> ScalarField:
        return self._collider_sdf

    @ti.kernel
    def update_fields(self , collider : ti.template() , implicit_surface : ti.template() ) :
        sdf , vf = ti.static(self._collider_sdf.field() , self._collider_vel.field())
        for I in ti.grouped(sdf):
            sdf[I] = min(sdf[I] , implicit_surface.sign_distance(I))
            vf[I] = collider.velocity_at(I)

# See "Fluid Engine Development" 3.4.1.2.1
# TODO : test in-place implementation
@ti.kernel
def extrapolate(
    input_grid   : ti.template(),
    output_grid  : ti.template(),
    in_marker   : ti.template(),
    out_marker  : ti.template()):

    input , output = ti.static(input_grid.field() , output_grid.field())
    valid0 , valid1 = ti.static(in_marker.field() , out_marker.field())
    sx , sy = ti.static(input.shape[0] , input.shape[1])

    for i , j in input:
        sum = output[i,j] * 0.0
        count = 0

        if valid0[i,j] == 0 :
            if i + 1 < sx and valid0[i+1 , j] == 1:
                sum += output[i+1, j]
                count +=1
            elif i > 0 and valid0[i-1,j] == 1:
                sum += output[i-1,j]
                count+=1
            elif j + 1 < sy and valid0[i ,j + 1] == 1:
                sum += output[i,j+1]
                count+=1
            elif j > 0 and valid0[i,j-1] == 1 :
                sum += output[i , j -1]
                count+=1

            if count > 0 :
                output[i,j] = sum / count
                valid1[i,j] = 1
        else :
            valid1[i,j] == 1

def extrapolate_to_region(
    input_grid : Grid, 
    output_grid : Grid , 
    marker : DataPair , 
    n_iter: int ):
    
    copy(input_grid.field() , output_grid.field())
    copy(marker.new.field() , marker.old.field())

    for _ in range(n_iter):
        extrapolate(input_grid , output_grid ,marker.old ,marker.new)
        marker.swap()

class Eulerian_Solver(GridMethod_Solver):
    def __init__(
        self ,
        size   : Tuple[int] ,
        velocity_pair       : DataPair,
        pressure_pair       : DataPair,
        advection_solver    : AdvectionSolver ,
        projection_solver   : ProjectionSolver ,
        diffusion_solver    : DiffusionSolver = None ,
        boundary_solver     : GridBoudaryConditionSolver = None
        ):

        self.advection_solver = advection_solver 
        self.projection_solver= projection_solver
        self.diffusion_solver = diffusion_solver
        self.boundary_solver = boundary_solver if boundary_solver is not None else STDBoundarySolver(size)

        self._velocity_pair = velocity_pair
        self._pressure_pair = pressure_pair

        self._marker = ti.field(dtype = ti.i32 , shape= size)
        self._marker2= ti.field(dtype = ti.i32 , shape= size)

        self._marker_pair = DataPair(self._marker , self._marker2 , MarkerField , size)

        self._extrapolate_depth = 5
        self._emitters = []
        self._colliders= []

        # TODO  simulate fluid with surface
        self.fluid_sdf = ConstantField(size , -1000.0)

    def velocity(self) -> VectorField:
        return self._velocity_pair.old

    def pressure(self) -> ScalarField:
        return self._pressure_pair.old

    def marker(self) -> MarkerField:
        return self._marker_pair.old

    ## ---------- setters --------------

    def set_advection_grids(self , grids: List[DataPair]):
        self.advection_grids = grids

    def set_externel_forces(self , forces : List[Callable[[float] , None]]) :
        self.force_calculators = forces

    def set_gravity(self , g ):
        self.gravity = ti.Vector(list(g))

    def add_emitter(self , emitter : GridEmitter):
        self._emitters.append(emitter)

    def add_collider(self , collider : Collider ):
        self._colliders.append(collider)

    def colliders(self ) -> List[Collider]:
        return self._colliders

    ## ---------- override ---------------

    def compute_advection(self, time_interval : float):
        # build marker
        self.mark_collider_region(self.marker().field())
        sdf = self.boundary_solver.collider_sdf()
        for pair in self.advection_grids :
            self.advection_solver.advect(self.velocity() , pair.old , pair.new , sdf ,time_interval)
            self.extropolateIntoCollider(pair)

        for pair in self.advection_grids :
            pair.swap()

        self.applyboundaryCondition()

    def compute_viscosity(self , time_interval : float):
        if self.diffusion_solver is not None :
            self.mark_fluid(
                self.marker().field() ,
                self.boundary_solver.collider_sdf() ,
                self.fluid_sdf )
            self.diffusion_solver.solve(
                self._velocity_pair.old ,
                self._velocity_pair.new, 
                self.marker() ,
                time_interval) 
            self._velocity_pair.swap()
            self.applyboundaryCondition()

    def compute_external_force(self , time_interval : float):
        for f in self.force_calculators:
            f(time_interval)
            self.applyboundaryCondition()

    def compute_projection(self , time_interval : float):
        self.projection_solver.projection(
            self.velocity() ,self._pressure_pair , 
            self._marker ,time_interval
        )
        self.applyboundaryCondition()

    def begin_time_intergrate(self, time_interval : float):
        for collider in self._colliders:
            collider.update(time_interval)

        for emitter in self._emitters:
            emitter.emit(time_interval)

        if self.boundary_solver is not None :
            self.boundary_solver.update_collider(self._colliders)

        self.applyboundaryCondition()

    def end_time_intergrate(self , time_interval : float):
        pass

    ## ------- for concrete implemetation ---------------

    @ti.kernel
    def mark_fluid(self , marker : ti.template() , boundary_sdf :ti.template() , fluid_sdf:ti.template()):
        for I in ti.grouped(marker):
            if insideSdf(boundary_sdf.value(I)) :
                marker[I] = FluidMark.Boundary
            elif insideSdf(fluid_sdf.value(I)):
                marker[I] = FluidMark.Fluid
            else :
                marker[I] = FluidMark.Air

    @ti.kernel
    def gravity_force(self , time_interval : Float):
        vf = ti.static(self.velocity().field())
        for I in ti.grouped(vf):
            vf[I] += self.gravity * time_interval

    @ti.kernel
    def mark_collider_region(self , marker : ti.template()):
        for I in ti.grouped(marker):
            marker[I] = 0 if self.boundary_solver.collider_sdf().value(I) < 0.0 else 1            

    def extropolateIntoCollider(self, pair : DataPair):
        # for collider
        pair.swap()
        extrapolate_to_region(pair.old , pair.new , self._marker_pair, self._extrapolate_depth)

    def applyboundaryCondition(self):
        if self.boundary_solver is not None :
            self.boundary_solver.constrain_velocity(self._velocity_pair , self._marker_pair ,self._extrapolate_depth )
