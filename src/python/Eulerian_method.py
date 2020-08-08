from fluid_solver import GridMethod_Solver
from abc import ABCMeta ,abstractmethod
from utils import DataPair , Grid , near_index
from typing import List , Callable , Union , Tuple , Optional 
from enum import Enum

import taichi as ti

# @ti.data_oriented
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

class Semi_Lagrangian(AdvectionSolver):

    class Order(Enum):
        RK_1 = 1
        RK_2 = 2
        RK_3 = 3

    def __init__(self , RK = 1):
        self.RK = RK

    @ti.func
    def backtrace(self ,  vec_grid , I , dt):
        p = I
        if ti.static(self.RK == Semi_Lagrangian.Order.RK_1):
            p -= dt * vec_grid.sample(I)
        elif ti.static(self.RK == Semi_Lagrangian.Order.RK_2):
            mid = p - 0.5 * dt * vec_grid.sample(I)
            p -= dt * vec_grid.sample(mid)
        elif ti.static(self.RK == Semi_Lagrangian.Order.RK_3):
            v1 = vec_grid.sample(I)
            p1 = p - 0.5 * dt * v1
            v2 = vec_grid.sample(p1)
            p2 = p - 0.75 * dt * v1
            v3 = vec_grid.sample(p2)
            p -= dt * ( 2/9 * v1 + 1/3 * v2 + 4/9 * v3)
        else :
            ti.static_print(f"unsupported order for RK{self.RK}")

        return p

    @ti.func
    def lin_interpolate(self , v1 , v2 , frac):
        return v1 + frac * (v2 - v1)

    @ti.func
    def bilin_interplolate2(self , grid , I):
        iu , iv = int(I[0]) , int(I[1])
        du , dv = I[0] - iu , I[1] - iv
        a = grid.sample(iu , iv)
        b = grid.sample(iu + 1 , iv)
        c = grid.sample(iu , iv + 1)
        d = grid.sample(iu + 1 , iv + 1)
        return self.lin_interpolate(
            self.lin_interpolate(a , b , du),
            self.lin_interpolate(c , d , du),
            dv )
    
    @ti.func
    def bilin_interplolate3(self, grid , I):
        iu , iv , iw = int(I[0]) , int(I[1]) , int(I[2])
        du , dv , dw = I[0] - iu , I[1] - iv , I[2] - iw
        a = grid.sample(iu , iv , iw) 
        b = grid.sample(iu + 1, iv , iw)
        c = grid.sample(iu , iv + 1, iw)
        d = grid.sample(iu + 1 , iv + 1 , iw)

        ua = grid.sample(iu , iv , iw + 1) 
        ub = grid.sample(iu + 1, iv , iw + 1)
        uc = grid.sample(iu , iv + 1, iw + 1)
        ud = grid.sample(iu + 1 , iv + 1 , iw + 1)

        low = self.lin_interpolate(
            self.lin_interpolate(a , b , du),
            self.lin_interpolate(c , d , du),
            dv )
        
        up = self.lin_interpolate(
            self.lin_interpolate(ua , ub , du),
            self.lin_interpolate(uc , ud , du),
            dv )

        return self.lin_interpolate(low , up , dw)

    @ti.func
    def binlinear_interpolate(self , grid , I):
        dim = ti.static(I.shape)
        ti.static_assert(dim == 2 or dim == 3 , "only 2d or 3d interpolate is supported now." )
        if ti.static(dim == 2):
            return self.bilin_interplolate2(grid, I)
        else:
            return self.bilin_interplolate3(grid, I)

    @ti.kernel
    def advect(self , vec_grid , in_grid , out_grid , dt):
        for I in vec_grid:
            pos = self.backtrace(vec_grid , ti.Vector(I) ,dt)
            out_grid[I] = self.binlinear_interpolate(in_grid , pos)

#TODO : bfacc
class BFACC(AdvectionSolver):

    def advect(self, vec_field , in_grid , out_grid , dt):
        pass

class ForwardEulerDeffusionSolver(DiffusionSolver):
    def __init__(self , diffusion_coefficient):
        self.coefficient = diffusion_coefficient
    
    # @ti.func
    # def build_marker(self):
    #     #TODO
    #     pass
    
    @ti.kernel
    def solve(
        self , 
        vec_field : Grid ,
        next_vec_field : Grid , 
        marker : Grid,
        dt : ti.f32 ):

        for I in vec_field.grid_data :
            next_vec_field[I] = \
                vec_field[I] + \
                self.coefficient * dt * self.laplacian(vec_field ,marker, I)

    @ti.func
    def laplacian( self , grid : Grid, marker : Grid , I : ti.template()) :
        resolution = grid.resolution()
        dfx , dfy = 0.0 , 0.0
        i , j = ti.static(I[0] , I[1])
        center = grid.sample(I)
        if i > 0 and marker.sample([i - 1,j]) == FluidMark.Fluid :
            dfx += grid.sample([i - 1, j]) - center
        elif i + 1 < resolution[0] and marker.sample([i+1 , j]) == FluidMark.Fluid :
            dfx += grid.sample([i + 1, j]) - center

        if j > 0 and marker.sample([ i , j -1 ]) == FluidMark.Fluid:
            dfy += grid.sample([i , j - 1]) - center
        elif j + 1 < resolution[1] and marker.sample([i , j+1]) == FluidMark.Fluid :
            dfy += grid.sample([i , j + 1]) - center
        
        #TODO : use marker and judge

        # return dfx / (dx ** 2) + dfy / (dy **2)
        return dfx + dfy # note that spacing of grid = ( 1, 1) as default now

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

    def set_advection_grids(self , grids: List[DataPair]):
        assert(isinstance(grids , list) , "type of grids should be List[DataPair].")
        self.advection_grids = grids
        if self._velocity_pair not in self.advection_grids :
            self.advection_grids.append(self.advection_grids)

    def set_externel_forces(self , forces : List[Callable[[float] , None]]) :
        assert(isinstance(forces , list) , "type of forces should be List[Callable[[float] , None]]")
        self.force_calculators = forces

    def compute_advection(self, time_interval : float):
        for pair in self.advection_grids :
            self.advection_solver.advect(self._velocity_pair.old , pair , time_interval)
            self.extropolateIntoCollider(pair)
        for pair in self.advection_grids :
            pair.swap()

        self.applyboundaryCondition()

    def extropolateIntoCollider(self, grid):
        #TODO extropolateIntoCollider
        pass

    def applyboundaryCondition(self):
        #TODO: apply boundary condition
        pass

    def compute_external_force(self , time_interval : float):
        for f in self.force_calculators:
            f(time_interval)    # update velocity field

    def compute_projection(self , time_interval : float):
        #TODO
        pass

    def build_marker(self):
        return self.marker

    def compute_viscosity(self , time_interval : float):
        if not self.diffusion_solver is None :
            marker = self.build_marker()
            self.diffusion_solver.solve(
                self._velocity_pair.old ,
                self._velocity_pair.new, 
                marker ,
                time_interval)      
