from fluid_solver import * 
from abc import ABCMeta ,abstractmethod
from utils import DataPair , Grid , Bilinear_Interp_Sampler , VectorField , ScalarField , clamp_index2
from typing import List , Callable , Union , Tuple , Optional 
from enum import Enum

from types import Float , Vector ,Index

import taichi as ti

class Semi_Lagrangian(AdvectionSolver):

    class Order(Enum):
        RK_1 = 1
        RK_2 = 2
        RK_3 = 3

    def __init__(self , RK = Semi_Lagrangian.Order.RK_1):
        assert(isinstance(RK , Semi_Lagrangian.Order) , "RK must be an instance of SemiLagrangian.Order.")
        self.RK = RK

    @ti.func
    def backtrace(self ,  vel_grid : VectorField, I : Index , dt : Float) -> Vector:
        p = int(I)
        if ti.static(self.RK == Semi_Lagrangian.Order.RK_1):
            p -= dt * vel_grid.value(I)
        elif ti.static(self.RK == Semi_Lagrangian.Order.RK_2):
            mid = p - 0.5 * dt * vel_grid.value(I)
            p -= dt * vel_grid.sample(mid)
        elif ti.static(self.RK == Semi_Lagrangian.Order.RK_3):
            v1 = vel_grid.value(I)
            p1 = p - 0.5 * dt * v1
            v2 = vel_grid.sample(p1)
            p2 = p - 0.75 * dt * v1
            v3 = vel_grid.sample(p2)
            p -= dt * ( 2/9 * v1 + 1/3 * v2 + 4/9 * v3)
        return p

    @ti.kernel
    def advect(
        self , 
        vel_grid : ti.template(),   # Grid (VectorField)
        in_grid : ti.template() ,   # Grid
        out_grid : ti.template() ,  # Grid
        dt : Float):

        out = ti.static(out_grid.field())

        for I in vel_grid.field():
            pos = self.backtrace(vel_grid , I ,dt)
            out[I] = in_grid.sample(pos)

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

        for I in vel_grid.field() :
            next_vel_grid[I] = vel_grid[I] + \
                self.coefficient * dt * self.laplacian(vel_grid ,marker, I)

    # TODO : reduce to gird.laplacian()
    @ti.func
    def laplacian( self , grid : Grid, marker : Grid , I : Index) :
        sz , ds = grid.size() , grid.spacing()
        dfx , dfy = 0.0 , 0.0
        i , j = ti.static(I[0] , I[1])
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
    def __init__(self , max_iter : int):
        self._max_iter = max_iter

    def projection(
        self ,
        vel_field : VectorField,
        pressure_pair : DataPair,
        density : ScalarField ,
        dt : Float ):

        for _ in range(self._max_iter) :
            self.jacobian_step(self , vel_field , density ,pressure_pair , dt) 
            pressure_pair.swap()

    @ti.kernel
    def jacobian_step(        
        self,
        vel_field : ti.template() ,
        density : ti.template() ,
        pressure_curr   : ti.template() , 
        pressure_next   : ti.template() ,
        dt : Float ):

        sz = ti.static(vel_field.size())
        for i , j in vel_field.field() :
            pl = pressure_curr.value(clamp_index2(i - 1 , j , sz))    
            pr = pressure_curr.value(clamp_index2(i + 1 , j , sz))
            pt = pressure_curr.value(clamp_index2(i , j + 1 , sz))
            pb = pressure_curr.value(clamp_index2(i , j - 1 , sz))
            pressure_next[i,j] = 0.25 *  (pl + pr + pt + pb - density[i,j] * vel_field.divergence([i,j]) / dt) 

class Eulerian_Solver(GridMethod_Solver):
    def __init__(
        self ,
        velocity_pair       : DataPair,
        pressure_pair       : DataPair,
        advection_solver    : AdvectionSolver ,
        projection_solver   : ProjectionSolver ,
        diffusion_solver    : DiffusionSolver = None
        ):
        
        self.advection_solver = advection_solver 
        self.projection_solver= projection_solver
        self.diffusion_solver = diffusion_solver

        self._velocity_pair = velocity_pair
        self._pressure_pair = pressure_pair

        # Compute marker
        self.marker = None

    @abstractmethod
    def density(self):
        pass

    @abstractmethod
    def velocity(self):
        pass

    @abstractmethod
    def pressure(self):
        pass

    ## ---------- setters --------------

    def set_advection_grids(self , grids: List[DataPair]):
        assert(isinstance(grids , list) , "type of grids should be List[DataPair].")
        self.advection_grids = grids
        if self._velocity_pair not in self.advection_grids :
            self.advection_grids.append(self.advection_grids)

    def set_externel_forces(self , forces : List[Callable[[float] , None]]) :
        assert(isinstance(forces , list) , "type of forces should be List[Callable[[float] , None]]")
        self.force_calculators = forces

    ## ---------- override ---------------

    def compute_advection(self, time_interval : float):
        for pair in self.advection_grids :
            self.advection_solver.advect(self._velocity_pair.old , pair , time_interval)
            self.extropolateIntoCollider(pair)
        for pair in self.advection_grids :
            pair.swap()

        self.applyboundaryCondition()

    def compute_external_force(self , time_interval : float):
        for f in self.force_calculators:
            f(time_interval)    # update velocity field

    def compute_projection(self , time_interval : float):
        #compute pressure 
        self.projection_solver.solve(
            self._velocity_pair.old ,# self._div_vel ,
            self._pressure_pair.old , self.density() ,time_interval
        )
        #apply boundary condition
        self.applyboundaryCondition()

    def begin_time_intergrate(self, time_interval : float):
        #TODO update collider 
        #TODO update emitter
        pass

    def end_time_intergrate(self , time_interval : float):
        #TODO update v
        pass     

    ## ------- for concrete implemetation ---------------

    def build_marker(self):
        #TODO
        return self.marker

    def viscosity_force(self , time_interval : float):
        if not self.diffusion_solver is None :
            marker = self.build_marker()
            self.diffusion_solver.projection(
                self._velocity_pair.old ,
                self._velocity_pair.new, 
                marker ,
                time_interval) 
            self._velocity_pair.swap()


    def extropolateIntoCollider(self, grid):
        #TODO extropolateIntoCollider
        # for collider
        pass

    def applyboundaryCondition(self):
        #TODO: apply boundary condition
        # No flux
        # Slip
        # check OpenBoundary => set velocity to be zero
        pass
