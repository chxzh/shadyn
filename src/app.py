import numpy as np
import glfw
from OpenGL.GL import *

class Application:
    def __init__(self):
        glfw.init()
        self.main_window = Window()
        self.window_list = [self.main_window]
        return
    
    def init(self):
        for window in self.window_list:            
            glfw.make_context_current(window.handle)
            window.init()
        return
    
    def terminate(self):
        glfw.terminate()
        return
    
    def loop_condition(self):
        return not glfw.window_should_close(self.main_window.handle)
    
    def main_loop(self):
        for window in self.window_list:
            glfw.make_context_current(window.handle)
            window.draw()
            glfw.swap_buffers(window.handle)
        glfw.poll_events()
        
    def run(self):
        self.init()
        while self.loop_condition():
            self.main_loop()
        self.terminate()
        return
        

class Window:
    def __init__(self, width=640, length=480, title="title"):
        self.width = width
        self.length = length
        self.title = title
        self.handle = glfw.create_window(width, length, title, None, None);
        self.background_color = (0.0, 0.0, 0.2, 1.0)
        return
    
    def init(self):
        glClearColor(self.background_color[0],
                     self.background_color[1],
                     self.background_color[2],
                     self.background_color[3])
        pass
    
    def draw(self):        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        return

class Scene:
    def __init__(self):
        return


class Item:
    def __init__(self):
        return

