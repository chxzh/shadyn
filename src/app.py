import numpy as np
import glfw
from OpenGL.GL import *
from cgkit.cgtypes import *
import math

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
    def __init__(self, item_list=[], camera=None):
        self.item_list=item_list
        self.camera = Camera() if camera == None else camera
        return
    
    def draw(self):
        pass


class Camera:
    def __init__(self,
                 position=vec3(0.),
                 target=vec3(0., 0., -1.),
                 up=vec3(0.,1.,0.),
                 projection_matrix=None,
                 static=False):
        self.position = position
        self.target = target
        self.view_matrix = mat4.lookAt(position, target, up)
        self.projection_matrix = projection_matrix\
            if not projection_matrix == None\
            else mat4.perspective(math.radians(45),4./3,0.1,100)
        
        pass
    


class Item:
    def __init__(self, obj, 
                 position=vec3(0.),
                 size=vec3(1.),
                 orientation=vec3(0.),
                 static=False):
        self.object = obj
        self.position = position
        self.size = size
        self.orientation = orientation
        self.translate_mat = mat4.translation(position)
        self.scale_mat = mat4.scaling(size)
        self.rotate_mat = mat4(1.)
        if not orientation == vec3(0.):
            # TODO: decide which euler roration system to apply
            pass                                         
        self.static = static
        return
    
    def translate(self, offset):
        self.translate_mat.translate(offset)
    
    def scale(self, magnitude):    
        self.scale_mat.scale(magnitude)
    
    def rotate(self, rad_angle, axis):
        self.rotate_mat.rotate(rad_angle, axis)
    
    def get_model_mat(self):
        return self.translate_mat*self.rotate_mat*self.scale_mat


class Object:
    def __init__(self, path=None):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.v_indice = []
        self.t_indice = []
        self.n_indice = []
        # if the path is not provided, presumably attributes will be filled in
        # afterward
        if path == None:
            return
        attrib_map = {'v': (self.vertices,3),
                     'vn': (self.normals, 3),
                     'vt': (self.texcoords, 2)}
        with open(path, 'r') as obj_file:
            for line in obj_file:
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] in attrib_map:
                    attrib, length = attrib_map[values[0]]
                    attrib.append(tuple(map(float, values[1:1+length])))
                elif values[0] == 'f':
                    face = []
                    for value in values[1:]:
                        w = map(lambda x: int(x) if x else None, value.split('/'))
                        # convert from 1-based to 0-based
                        w = map(lambda x: x - 1 if x != None and x > 0 else x, w)
                        # fill the values into corresponding list
                        # TODO: decide what should be filled in for missing value
                        # (which is None currently filled in)
                        v, vt, vn = tuple(w)
                        self.v_indice.append(v)
                        self.t_indice.append(vt)
                        self.n_indice.append(vn)
                        face.append(tuple(w))
                    self.faces.append(tuple(face))
        return
    
def _main():
    pass

if __name__ == "__main__":
    print "---starting main()---"
    _main()
    print "---end of main()---"
