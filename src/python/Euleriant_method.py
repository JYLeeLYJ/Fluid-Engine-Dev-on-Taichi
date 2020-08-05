from fluid_solver import GridMethod_Solver
from abc import ABCMeta ,abstractmethod
from utils import DataPair , Grid
from typing import List , Callable , Union , Tuple , Optional 
from enum import Enum

import taichi as ti

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
    def __init__(self):
        self.marker = None
    
    @ti.func
    def build_marker(self):
        #TODO
        pass
    
    @ti.kernel
    def solve(
        self , 
        vec_field : Grid ,
        next_vec_field : Grid , 
        diffusionCoefficient : ti.f32 , 
        dt : ti.f32 ):

        for I in vec_field.grid_data :
            next_vec_field[I] = \
                vec_field[I] + \
                diffusionCoefficient * dt * self.laplacian(vec_field , self.marker , I)

    @ti.func
    def laplacian( self , grid : Grid, marker : Grid , I : List) :
        dim = ti.static(len(I))
        res = 0.0
        for i in range(dim):
            #TODO : use marker and judge
            pass
        return res

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
        if self._velocity_pair not in self.advection_grids :
            self.advection_grids.append(self.advection_grids)

    def set_externel_forces(self , forces : List[Callable[[float] , None]]) :
        assert(isinstance(forces , list) , "type of forces should be List[Callable[[float] , None]]")
        self.force_calculators = forces

    def compute_advection(self, time_interval : float):
        for pair in self.advection_grids :
            self.advection_solver.advect(self._velocity_pair.old , pair , time_interval)
        for pair in self.advection_grids :
            pair.swap()

        self.applyboundaryCondition()

    def applyboundaryCondition(self):
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
