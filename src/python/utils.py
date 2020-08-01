from typing import Union ,Tuple
from abc import ABCMeta , abstractmethod

import taichi as ti

Ti_Field = Union [ti.Matrix , ti.Expr]
Ti_Matrix = Ti_Field
Ti_Vector = Ti_Matrix
Ti_Scalar = Union[ti.f32 , ti.f64 , ti.i8 , ti.i16 , ti.i64 , ti.i32 ]


class GridSampler(metaclass = ABCMeta):
    @abstractmethod
    def sample(self , grid : Ti_Field , pos : Ti_Vector ) -> Union[Ti_Vector, Ti_Scalar]:
        pass

@ti.data_oriented
class Grid:
    def __init__(
        self , 
        grid : Ti_Field , 
        sampler : GridSampler ,
        spacing : Union[Tuple[int , int, int ] , Tuple[int , int] , int] = 1
        ):

        self.grid_data = grid   
        self.sampler = sampler 

    @ti.func
    def resolution(self) -> Ti_Vector:
        return ti.Vector(list(self.grid_data.shape))

    # @ti.func
    # def origin(self):
    #     pass

    @ti.func
    def grid_spacing(self):
        #TODO
        pass

    @ti.func
    def bounding_box(self):
        #TODO
        pass

    @ti.func
    def sample(self , pos : Ti_Vector ) -> Union[Ti_Vector , Ti_Scalar]:
        return self.sampler(self.grid_data , pos)

class DirectlySampler(GridSampler):
    @ti.func
    def sample(self , grid : Ti_Field , pos : Ti_Vector ) -> Union[Ti_Vector, Ti_Scalar]:
        return grid[pos]        

class DataPair:
    def __init__(self , old : Ti_Field , new : Ti_Field , sampler : GridSampler):
        self.new = Grid(new , sampler) 
        self.old = Grid(old , sampler)

    def swap(self):
        self.new , self.old = self.old , self.new 

