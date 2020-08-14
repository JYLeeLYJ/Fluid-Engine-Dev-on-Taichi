from typing import Union ,Tuple , List
from abc import ABCMeta , abstractmethod

from basic_types import Index , Vector , Matrix , Float , Int
from helper_functions import clamp_index2 ,linear_interpolate
import taichi as ti

class DataPair:
    def __init__(self , old  , new , Grid ):
        self.new = Grid(new) 
        self.old = Grid(old)

    def swap(self):
        self.new , self.old = self.old , self.new 

@ti.data_oriented
class Grid(metaclass = ABCMeta ):

    def __init__( 
        self , 
        size : Tuple[int] or List[int] ,
        spacing : Tuple[int] or List[int] = (1,1)):

        assert len(spacing) == 2 and len(size) == 2
        
        self._spacing = list(spacing)
        self._size = list(size)

    def bounding_box(self):
        #TODO implement bounding box 
        pass

    @abstractmethod
    def sample(self , pos : Vector) -> Vector or Float :
        pass

    @abstractmethod
    def value(self , pos : Index) -> Vector or Float:
        pass

    @abstractmethod
    def zero_value(self) :
        pass

    @abstractmethod
    def one_value(self):
        pass

    # @ti.func
    def size(self) -> Vector :
        # return ti.Vector(self._size)
        return self._size

    # @ti.func
    def spacing(self) -> Vector :
        # return ti.Vector(self._spacing)
        return self._spacing

class Sampler(metaclass = ABCMeta):
    @abstractmethod
    def sample_value(self , grid : Grid, pos : Index ) -> Vector or Float :
        pass

# Bilinear interpolation sampling in 2d grids
@ti.data_oriented
class Bilinear_Interp_Sampler(Sampler):
    def __init__(self , size):
        self._size = size

    @ti.func
    def sample_value(self , grid : ti.template() , pos : Vector) -> ti.template():   # Vector or Scalar
        s , t = pos[0] - 0.5 , pos[1] - 0.5
        iu , iv = int(s) , int(t)
        du , dv = s - iu , t - iv
        a = grid[clamp_index2(int(iu + 0.5) , int(iv + 0.5) , self._size)]
        b = grid[clamp_index2(int(iu + 1.5) , int(iv + 0.5) , self._size)]
        c = grid[clamp_index2(int(iu + 0.5) , int(iv + 1.5) , self._size)]
        d = grid[clamp_index2(int(iu + 1.5) , int(iv + 1.5) , self._size)]
        return linear_interpolate(
            linear_interpolate( a , b , du) , 
            linear_interpolate( c , d , du) , 
            dv
        )

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
    def sample(self,pos : Vector) -> ti.template():
        return self._value

    @ti.func
    def value(self, pos : Index) -> ti.template():
        return self._value

    """ zero and one are meaningless in this class"""
    @ti.func
    def zero_value(self):
        return self._value
    
    @ti.func
    def one_value(self):
        return self._value

@ti.data_oriented
class MarkerField(Grid):
    def __init__( self , marker, spacing = (1,1) , sampler = None):
        self.marker = marker
    
    @ti.func
    def sample(self , pos : Vector) -> Int :
        return self.marker[int(pos)]

    @ti.func
    def value(self , pos : Index) -> Int:
        return self.marker[pos]

    @ti.func
    def zero_value(self) -> Int:
        return 0

    @ti.func
    def one_value(self) -> Int:
        return 1

@ti.data_oriented
class ScalarField(Grid):
    def __init__( self , ti_field, spacing = (1,1) , sampler = None):
        super().__init__(ti_field.shape , spacing)

        self._grid = ti_field
        self._sampler = Bilinear_Interp_Sampler(ti_field.shape) if sampler == None else sampler

    @ti.func
    def sample(self, pos : Vector)->Float:
        return self._sampler.sample_value(self._grid, pos)

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
    def zero_value(self) -> Float:
        return 0.0

    @ti.func
    def one_value(self) -> Float :
        return 1.0

    @ti.func
    def laplacian(self , pos : Index) -> Float :
        sz , ds = self.size() , self.spacing()
        i , j = pos[0] , pos[1]

        centre  =  self._grid[pos]
        left = self._grid[clamp_index2(i - 1 , j , sz)]
        right= self._grid[clamp_index2(i + 1 , j , sz)]
        top  = self._grid[clamp_index2(i , j + 1 , sz)]
        down = self._grid[clamp_index2(i , j - 1 , sz)]

        return (right + left - 2 * centre) / (ds[0] ** 2) + (top + down - 2 * centre) / (ds[1] ** 2)

    def field(self) -> ti.template():   # ScalarField
        return self._grid

@ti.data_oriented
class VectorField(Grid):
    def __init__(self , ti_field , spacing = (1,1) , sampler = None):
        super().__init__(ti_field.shape , spacing)

        self._grid = ti_field
        self._sampler = Bilinear_Interp_Sampler(ti_field.shape) if sampler == None else sampler

    @ti.func
    def sample(self, pos : Vector )->Vector:
        return self._sampler.sample_value(self._grid, pos)

    @ti.func
    def value(self, pos : Index)->Float:
        return self._grid[pos]

    @ti.func
    def one_value(self) -> Vector:
        return ti.Vector.one(ti.f32 , 2)
    
    @ti.func
    def zero_value(self) -> Vector:
        return ti.Vector.zero(ti.f32 ,2)

    @ti.func
    def divergence(self, pos : Index) -> Float:
        sz , ds = self.size() , self.spacing()
        i , j = pos[0] , pos[1]

        left = self._grid[clamp_index2(i - 1 , j , sz)][0]
        right= self._grid[clamp_index2(i + 1 , j , sz)][0]
        top  = self._grid[clamp_index2(i , j + 1 , sz)][1]
        down = self._grid[clamp_index2(i , j - 1 , sz)][1]

        return 0.5 * ((right - left) / ds[0] + (top - down)/ds[1])

    def field(self) -> ti.template():   # VectorField
        return self._grid
