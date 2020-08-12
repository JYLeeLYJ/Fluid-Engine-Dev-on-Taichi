from animation import Animation
from smoke_solver import build_smoke
import taichi as ti

@ti.data_oriented
class Smoke_Animation(Animation):
    def __init__(self , resolution = (512,512), time_interval = 0.04):
        import numpy as np

        self.dt = time_interval
        self.pixels = ti.Vector.field(3 ,dtype = ti.f32 , shape = resolution)
        self.dcolor = ti.Vector(list(np.random.rand(3) * 0.7 + 0.3) )

        self.smoke = build_smoke(resolution)

    def reset(self):
        self.smoke.reset()
        self.pixels.fill([0,0,0])

    @ti.kernel
    def render(self):
        d = ti.static(self.smoke.density().field())
        px = ti.static(self.pixels)
        for I in ti.grouped(px):
            px[I] = d[I] * self.dcolor

    def update(self):
        self.smoke.advance_time_step(self.dt)

    def display(self , gui : ti.GUI):
        self.render()
        gui.set_image(self.pixels)
        gui.show()
