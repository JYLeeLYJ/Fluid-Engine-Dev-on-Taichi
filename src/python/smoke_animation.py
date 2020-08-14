from animation import Animation
import taichi as ti
# from gif_maker import GIF_Maker

@ti.data_oriented
class Smoke_Animation(Animation):
    def __init__(self , solver , resolution = (512,512), time_interval = 0.04):
        import numpy as np

        self.dt = time_interval
        self.pixels = ti.Vector.field(3 ,dtype = ti.f32 , shape = resolution)
        self.dcolor = ti.Vector(list(np.random.rand(3) * 0.7 + 0.3))
        
        self.bound_color = ti.Vector([255.0 ,99.0 , 71.0]) / 255.0
        self.smoke = solver

        # self.gif = GIF_Maker()

    def reset(self):
        self.smoke.reset()
        self.pixels.fill([0,0,0])

    @ti.kernel
    def render(self):
        d = ti.static(self.smoke.density().field())
        px = ti.static(self.pixels)
        sdf = ti.static(self.smoke.boundary_solver.collider_sdf().field())
        for I in ti.grouped(px):
            if sdf[I] < -4.0 :
                px[I] = self.bound_color * 0.7     # collider inside color
            elif sdf[I] < 0.0 :
                px[I] = self.bound_color    # collider boundary color
            else:
                px[I] = d[I] * self.dcolor  # smoke color

    def update(self):
        self.smoke.advance_time_step(self.dt)

    def display(self , gui : ti.GUI):

        # if gui.get_event(ti.GUI.PRESS):
        #     if gui.is_pressed(ti.GUI.SPACE) : 
        #         self.gif.change()

        self.render()
        # self.gif.set_img(self.pixels)
        gui.set_image(self.pixels)
        gui.show()
