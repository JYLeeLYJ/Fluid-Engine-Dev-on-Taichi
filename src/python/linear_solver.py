
from abc import ABCMeta , abstractmethod

from utils import DataPair

# @class Vector
class Vector(metaclass = ABCMeta):
    def __init__(self):
        pass

# @class Matrix
class Matrix(metaclass = ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, x : Vector , b : Vector , result : Vector):
        pass

# @class Linear_Solver
class Linear_Solver (metaclass = ABCMeta):
    def __init__(self ) : 
        pass

    @abstractmethod
    def solve(self , x , x_new , b ):
        pass

import taichi as ti

class STDJacobian_Mat(Matrix):

    def __init__(self , A :Matrix):
        self.A = A

    @ti.func
    def row_sum(self , x  , A , I ) :
        res = 0.0
        for J in ti.grouped(x) :
            res -= A[I , J] * x [J] if I != J else 0.0
        return res / A[I,I]

    @ti.kernel
    def apply(self , x : ti.template() , b : ti.template() ,result : ti.template()):
        for I in ti.grouped(x) :
            result[I] = self.row_sum( x , self.A , I)

class Jacobian_Iterate(Linear_Solver):
    def __init__(self ,  n : int,  A : Matrix , max_iter : int = 30 , jacobian_mat : Matrix = None  ) :
        self.max_iter = max_iter
        self.jacobi_mat = STDJacobian_Mat(A) if jacobian_mat == None else jacobian_mat
        pass

    def solve(self , x : Vector, x_new : Vector , b : Vector) :
        pair = DataPair(x , x_new)
        for _ in range(self.max_iter):
            x_new.fill(0.0)
            self.jacobi_mat.apply(pair.old , b , pair.new) # jacobi step
            pair.swap()



class PreConditioner(metaclass = ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def apply_preconditioner(self):
        pass


class ConjugateGradient_Slover(Linear_Solver):
    def __init__(self , A : Matrix , n : int , max_iter : int = 20):
        pass

    def solve(self , x):
        pass