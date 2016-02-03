import os, plotting, log
from shadow_optim import Renderer, Optimizer
from datetime import datetime as dt
def get_fname(root=None):
    if not root:
        root = os.getcwd()
    elif not os.path.exists(root):
        os.mkdir(root)
    time_stamp_str = dt.now().strftime("%m-%d-%H-%M-%S-%y")
    path = root + '\\' + time_stamp_str
    if not os.path.exists(path):
        os.mkdir(path)
    print path
    return path + '\\' + time_stamp_str

def vanilla():
    renderer = Renderer()
    renderer.start()
    renderer.wait_till_init()
    plotter = plotting.Plotter(get_fname("..\\res"))
    optimizer = Optimizer(renderer)
    plotting.attach_plotter(optimizer, plotter)
    optimizer.set_target("C:\\Users\\cxz\\Pictures\\target.png")
    optimizer.set_method("Powell")
    optimizer.set_energy(["first moments (normalized)"], [1])
    optimizer.run()

def main():
#     os.mkdir("..\\res\\now")
    vanilla()
    pass

if __name__ == "__main__":
    print "before main"
    main()
    print "after main"