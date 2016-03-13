import os, plotting, log
from shadow_optim import Optimizer
# from shadow_optim import Renderer
from rendering import Renderer
from datetime import datetime as dt
from itertools import product
from tools import get_fname
from cal import *
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
    renderer.set_energy_terms(Optimizer.energy_dic.keys())
    from PIL import Image
    renderer.set_target_image(Image.open("..\\img\\target_mickey.png").convert('L'))
    renderer.start()
    renderer.wait_till_init()
    import shadow_optim
    init_X_Y(*renderer.viewport_size)
    plotter = plotting.Plotter(*get_fname("..\\res"))
    optimizer = Optimizer(renderer)
    plotting.attach_plotter(optimizer, plotter)
    optimizer.set_target("..\\img\\target_mickey.png")
    optimizer.set_method("CMA")
#     optimizer.set_energy(["first moments (normalized)"], [1])
    optimizer.set_energy(["first moments (normalized)", "XOR comparison"], 
                         [1,  1])
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