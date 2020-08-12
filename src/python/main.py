import taichi as ti
from smoke_animation import Smoke_Animation


res = (512 ,512)

ti.init(arch = ti.gpu , kernel_profiler = True)
gui = ti.GUI("smoke animation" , res = res)

ani = Smoke_Animation(res)
ani.reset()

while gui.running:
    ani.update()
    ani.display(gui)

ti.kernel_profiler_print()