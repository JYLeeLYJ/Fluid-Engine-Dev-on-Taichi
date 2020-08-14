import taichi as ti
from smoke_solver import Smoke_Builder , FlowEmitter
from smoke_animation import Smoke_Animation
from colliders import RigidBodyCollier , Collider
from geometry import Box , Ball

res = (512 ,512)

ti.init(arch = ti.gpu , kernel_profiler = True)
gui = ti.GUI("smoke animation" , res = res)

# build smoke solver
smoke = \
    Smoke_Builder(res)  \
    .add_flow_emitter([512//2 , 0] , 512//3 , 2000.0)    \
    .set_decay(0.995)   \
    .build()

    # .set_compute_buoyancy_force(tempreture_factor=600)\

ani = Smoke_Animation(smoke ,res)
ani.reset()

# collider
smoke.add_collider(RigidBodyCollier(Box([156,156] , [226 , 276])))
smoke.add_collider(RigidBodyCollier(Ball([276,226] , 30)))

while gui.running:
    ani.update()
    ani.display(gui)

ti.kernel_profiler_print()