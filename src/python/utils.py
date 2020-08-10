from typing import Union ,Tuple , List
from abc import ABCMeta , abstractmethod
from types import Index , Vector , Matrix , Float , Int

import taichi as ti

### ================ Utility Functions =============================

@ti.func
def clamp_index( I : Index , size : Vector) -> Index:
    # dim = ti.static(len(I.shape))
    # index = []
    # for i in range(dim):
    #     index.append(max(0,min(I[i] , size[i])))
    # return index

    ti.static_assert(len(size.shape) == 2 and len(I.shape) == 2)
    i = max(0,min(int(I[0]) ,size[0] -1))
    j = max(0,min(int(I[1]) ,size[1] -1))
    return ti.Vector([i,j])

@ti.func
def clamp_index2( x : Int , y : Int , size : Vector) -> Index :
    i = max(0,min(x ,size[0] -1))
    j = max(0,min(y ,size[1] -1))
    return ti.Vector([i,j])

@ti.func
def linear_interpolate( v1 : ti.template(), v2 : ti.template()  , fraction : Float) -> ti.template() :
    return v1 + fraction * (v2 - v1)

### ================ Utility types ================================

class DataPair:
    def __init__(self , old  , new , Grid ):
        self.new = Grid(new) 
        self.old = Grid(old)

    def swap(self):
        self.new , self.old = self.old , self.new 


@ti.data_oriented
class Grid(metaclass = ABCMeta ):

    class Sampler(metaclass = ABCMeta):
        @abstractmethod
        def sample_value(self , grid : Grid, pos : Index ) -> Vector or Float :
            pass

    def __init__( 
        self , 
        size : Tuple[int] or List[int] ,
        spacing : Tuple[int] or List[int] = (1,1)):

        assert len(spacing) == 2 and len(size) == 2
        
        self._spacing = spacing
        self._size = size

    def bounding_box(self):
        #TODO implement bounding box 
        pass

    @abstractmethod
    def sample(self , pos : Index, sampler : Grid.Sampler) -> Vector or Float :
        pass

    @abstractmethod
    def value(self , pos : Index) -> Vector or Float:
        pass

    @ti.func
    def size(self) -> Vector :
        return ti.Vector(self.size)

    @ti.func
    def spacing(self) -> Vector :
        return ti.Vector(self.spacing)

@ti.data_oriented
class ConstantField(Grid):
    def __init__(
        self , 
        size : Tuple[int] or List[int], 
        const_value : Float or Vector , 
        spacing : Tuple[int] or List[int] = (1,1)):
        
        self._value = const_value
        super().__init__(size , spacing)

    @ti.func
    def sample(self,pos : Index , sampler : Grid.Sampler):
        return self._value

    @ti.func
    def value(self, pos : Index):
        return self._value

@ti.data_oriented
class ScalarField(Grid):
    def __init__( self , ti_field, spacing = (1,1)):
        self._grid = ti_field
        super().__init__(self._grid.shape , spacing)

    @ti.func
    def sample(self, pos : Index , sampler : Grid.Sampler)->Float:
        return sampler.sample_value(self._grid, pos)

    @ti.func
    def value(self, pos : Index)->Float:
        return self._grid[pos]

    @ti.func
    def gradient(self , pos : Index) -> Vector :
        sz , ds = ti.static(self.size() , self.spacing())
        i , j = pos[0] ,pos[1]
        left = self._grid[clamp_index2(i - 1 , j , sz)]
        right= self._grid[clamp_index2(i + 1 , j , sz)]
        top  = self._grid[clamp_index2(i , j + 1 , sz)]
        down = self._grid[clamp_index2(i , j - 1 , sz)]

        return  0.5 * ti.Vector([(right - left) / ds[0] , (top - down) / ds[1]])

    @ti.func
    def laplacian(self , pos : Index) -> Float :
        sz , ds = self.size() , self.spacing()
        i , j = pos[0] , pos[1]

        centre  =  self._grid[pos]
        left = self._grid[clamp_index2(i - 1 , j , sz)]
        right= self._grid[clamp_index2(i + 1 , j , sz)]
        top  = self._grid[clamp_index2(i , j + 1 , sz)]
        down = self._grid[clamp_index2(i , j - 1 , sz)]

        return (right + left - 2 * centre) / ds[0] ** 2 + (top + down - 2 * centre) / ds[1] ** 2

@ti.data_oriented
class VectorField(Grid):
    def __init__(self , ti_field , spacing = (1,1)):
        self._grid = ti_field
        super().__init__(self._grid.shape , spacing)

    @ti.func
    def sample(self, pos : Index , sampler : Grid.Sampler)->Vector:
        return sampler.sample_value(self._grid, pos)

    @ti.func
    def value(self, pos : Index)->Float:
        return self._grid[pos]

    @ti.func
    def divergence(self, pos : Index) -> Float:
        sz , ds = self.size() , self.spacing()
        i , j = pos[0] , pos[1]

        left = self._grid[clamp_index2(i - 1 , j , sz)][0]
        right= self._grid[clamp_index2(i + 1 , j , sz)][0]
        top  = self._grid[clamp_index2(i , j + 1 , sz)][1]
        down = self._grid[clamp_index2(i , j - 1 , sz)][1]

        return 0.5 * ((right - left) / ds[0] + (top - down)/ds[1])

# Bilinear interpolation sampling in 2d grids
@ti.data_oriented
class Bilinear_Interp_Sampler(Grid.Sampler):
    def __init__(self):
        pass

    @ti.func
    def sample_value(self , grid : Grid , pos : Vector) -> ti.template():   # Vector or Grid
        I = ti.static(pos)
        iu , iv = int(I[0]) , int(I[1])
        du , dv = I[0] - iu , I[1] - iv
        a = grid[iu , iv]
        b = grid[iu + 1 , iv]
        c = grid[iu , iv + 1]
        d = grid[iu + 1 , iv + 1]
        return linear_interpolate(
            linear_interpolate( a , b , du) , 
            linear_interpolate( c , d , du) , 
            dv
        )

# @ti.data_oriented
# class ClampSampler(GridSampler):
#     def __init__(self , resolution):
#         self.resolution = resolution

#     @ti.func
#     def sample(self , field , Indices):
#         dim = ti.static(len(self.resolution))
#         I = []
#         for i in range(dim):
#             I.append(max(0, min(self.resolution[i] - 1, int(self.resolution[i]))))
#         return field[I]

# @ti.data_oriented
# class DirectlySampler(GridSampler):
#     @ti.func
#     def sample(self , grid , pos ):
#         # might lead to undefined behavior if pos is out of index boundary
#         return grid[pos]    

# @ti.data_oriented
# class ConstantSampler(GridSampler):

#     def __init__(self , const_value):
#         self._value = const_value

#     @ti.func
#     def sample(self , grid , pos):
#         return self._value

