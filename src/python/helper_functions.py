import taichi as ti
from basic_types import  Vector , Index , Float ,Int

@ti.func
def clamp( val , low , high):
    return max(low , min(val , high))

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

@ti.func
def insideSdf(phi : Float) :
    return phi < 0.0

@ti.func
def fraction_insideSdf(phi0 : Float , phi1 : Float) -> Float:
    ret = 0.0
    if insideSdf(phi0) and insideSdf(phi1) :
        ret = 1.0
    elif insideSdf(phi0) and not insideSdf(phi1):
        ret = phi0 / (phi0 - phi1)
    elif not insideSdf(phi0) and insideSdf(phi1):
        ret = phi1 / (phi1 - phi0)
    return ret
