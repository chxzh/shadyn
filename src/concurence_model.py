from threading import Thread, Lock
from bokeh.mplexporter.renderers.base import Renderer
from time import sleep
from random import randint
from __builtin__ import str
from colorama import Fore, Style
console_lock = Lock()

def get_print(title, color):
    def print_(s):
        console_lock.acquire()
        print color+Style.BRIGHT+"[%s]"%title, str(s) + Style.RESET_ALL
        console_lock.release()
    return print_

print_ = get_print('DEB', Fore.WHITE)
class Optimizer(Thread):
    def __init__(self, renderer):
        Thread.__init__(self)
        self.renderer = renderer
        self.print_ = get_print("Opt", Fore.GREEN) 
    
    def optimize(self):
        while True:
            x = randint(0,1000)
            self.print_("param %d"%x)
            self.renderer.param_update.acquire()
            self.renderer.set_param(x)
            self.renderer.param_update.release()
#             self.renderer.param_ready.release()
            # get snapshoot
            self.renderer.ss_update.acquire()
            self.renderer.ss_ready.acquire()
            self.renderer.ss_update.release()
            self.print_("get snapshot - %d" % self.renderer.snapshot)
            # compute
            sleep(randint(0,20)/10.)
        pass
    
    def run(self):
        self.optimize()


class Renderer(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.x = 0
        self.snapshot = 0
        self.param_update = Lock()
        print_(self.param_update.locked())
        self.param_ready = Lock()
        self.param_ready.acquire() # not ready
        self.ss_update = Lock() # not wanted
        self.ss_ready = Lock()
        self.ss_ready.acquire() # not ready
        self.param_lock = Lock()
        self.print_ = get_print("Rnd", Fore.YELLOW) 

    def set_param(self, x):
        self.x = x
        
    def draw(self):
        sleep_time = randint(0,5)/10.
        self.print_("drawing")
#         if self.param_update.locked():
#             # if it is updating, wait until it is ready
#             self.param_ready.acquire()
        self.param_update.acquire()
        self.print_("using param - %d" % self.x)
        sleep(sleep_time) # rendering to frame buffer
        self.snapshot = self.x # swap frame buffer
        self.print_("frame buffer - %d" % self.snapshot)
        if self.ss_update.locked():
            self.save_snapshot()
            self.ss_ready.release()
        self.param_update.release()
        sleep(0.5 - sleep_time)
    
    def save_snapshot(self):
        self.print_("saving snapshot - %d - %d"%(self.snapshot, self.x))
        
    def run(self):
        while True:
            self.draw()


if __name__ == "__main__":
    renderer = Renderer()
    optimizer = Optimizer(renderer)
    renderer.start()
    optimizer.start()
    