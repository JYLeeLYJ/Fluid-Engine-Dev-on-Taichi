from typing import Union ,Tuple
from abc import ABCMeta , abstractmethod

import taichi as ti

@ti.func
def near_index( I , dim   , distance ) :
    ti.static_assert( dim == len(I) , "dimension mismatch.")
    res = ti.Vector(I)
    res[dim] += distance
    return res

class GridSampler(metaclass = ABCMeta):
    @abstractmethod
    def sample(self , grid , pos  ) :
        pass

@ti.data_oriented
class Grid:
    def __init__(
        self , 
        grid ,
        sampler ,
        # spacing = 1,
        ):

        self.grid_data = grid   
        self.sampler = sampler 

    @ti.func
    def resolution(self) :
        return ti.Vector(list(self.grid_data.shape))

    @ti.func
    def bounding_box(self):
        #TODO
        pass

    @ti.func
    def sample(self , pos):
        return self.sampler(self.grid_data , pos)

class ClampSampler(GridSampler):
    def __init__(self , resolution):
        self.resolution = resolution

    @ti.func
    def sample(self , field , Indices):
        dim = ti.static(len(self.resolution))
        I = []
        for i in range(dim):
            I.append(max(0, min(self.resolution[i] - 1, int(self.resolution[i]))))
        return field[I]

class DirectlySampler(GridSampler):
    @ti.func
    def sample(self , grid , pos ):
        return grid[pos]    # might lead to undefined behavior if pos is out of index boundary

class DataPair:
    def __init__(self , old  , new , sampler ):
        self.new = Grid(new , sampler) 
        self.old = Grid(old , sampler)

    def swap(self):
        self.new , self.old = self.old , self.new 

