import os, plotting, log
from shadow_optim import Optimizer
# from shadow_optim import Renderer
from rendering import Renderer
from datetime import datetime as dt
from itertools import product
def get_fname(root=None):
    if not root:
        root = os.getcwd()
    elif not os.path.exists(root):
        os.mkdir(root)
    time_stamp_str = dt.now().strftime("%m-%d-%H-%M-%S-%y")
    path = root + '\\' + time_stamp_str
    if not os.path.exists(path):
        os.mkdir(path)
    return path, time_stamp_str

def get_renderer():
    renderer = Renderer()
    renderer.start()
    renderer.wait_till_init()
    return renderer

def get_default_plotted_optimizer(renderer):    
    optimizer = Optimizer(renderer)
    plotting.attach_plotter(optimizer, plotting.Plotter(*get_fname("..\\res")))
    return optimizer
    
def single_energy_combo(method_name, energy_name):
    renderer = get_renderer()
    optimizer = get_default_plotted_optimizer(renderer)
    optimizer.set_target("C:\\Users\\cxz\\Pictures\\target.png")
    optimizer.set_method(method_name)
    optimizer.set_energy([energy_name], [1])
    optimizer.run()
    
def all_single_energy_combos():
    renderer = Renderer()
    renderer.start()
    renderer.wait_till_init()
    x_0 = renderer.get_param()
    for method_name, energy_name in product(Optimizer.method_dic, Optimizer.energy_dic):
        print method_name, energy_name
        renderer.set_param(x_0)
        optimizer = Optimizer(renderer)
        plotting.attach_plotter(optimizer, plotting.Plotter(*get_fname("..\\res")))
        optimizer.set_target("C:\\Users\\cxz\\Pictures\\target.png")
        optimizer.set_method(method_name)
        optimizer.set_energy([energy_name], [1])
        optimizer.run()
    
def vanilla():
    renderer = Renderer()
    renderer.start()
    renderer.wait_till_init()
    import shadow_optim
    shadow_optim._init_X_Y(renderer.window.width, renderer.window.height)
    plotter = plotting.Plotter(*get_fname("..\\res"))
    optimizer = Optimizer(renderer)
    plotting.attach_plotter(optimizer, plotter)
    optimizer.set_target("C:\\Users\\cxz\\Pictures\\target_mickey.png")
    optimizer.set_method("CMA")
    optimizer.set_energy(["first moments (normalized)"], [1])
#     optimizer.set_energy(["first moments (normalized)", "XOR comparison"], 
#                          [1,  1])
#     optimizer.set_energy(["XOR comparison"], [1])
    optimizer.run()
    
def testing_moments():
    from PIL import Image
    import shadow_optim
    im = Image.open("C:\\Users\\cxz\\Pictures\\target_mickey.png")
    im = im.convert('L')
    print shadow_optim._get_fst_moments(im)

def main():
#     single_energy()
    vanilla()
#     single_energy_combo("Powell", "XOR comparison")
#     testing_moments()
    pass

if __name__ == "__main__":
    print "before main"
    main()
    print "after main"