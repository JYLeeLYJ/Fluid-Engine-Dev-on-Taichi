import taichi as ti

Float = ti.f32
Int = ti.i32
# Bool = ti.i32   # 0 is False , else is True

Vector = ti.template()
Matrix = ti.template()

Index = Vector

# ScalarGrid = Grid
# VectorGrid = Grid

INT_TYPEs = [int , ti.i8 , ti.i16 , ti.i32 , ti.i64]
FLAOT_TYPEs = [float , ti.f32 , ti.f64]

def is_int_type(dtype):
    return dtype in INT_TYPEs

def is_float_type(dtype):
    return dtype in FLAOT_TYPEs