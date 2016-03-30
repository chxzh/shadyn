import numpy as np
import glfw
from OpenGL.GL import *
from cgkit.cgtypes import *
from math import pi, sin, cos, tan
import math
from cgkit.lookat import LookAt

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
                 position=None,
                 orientation=None,
                 up=None,
                 projection_matrix=None,
                 resolution=None):
        self.position = position if position else vec3(0,0,0)
        self.orientation = orientation.normalize() if orientation and orientation.length() else vec3(0., 0., -1.)
        self.up = up if up and up.length() else vec3(0.,1.,0.)
        self.view_mat = self.look_at(self.position, self.position + self.orientation, self.up)
        self.resolution = resolution if resolution else (640,480)
        self.projection_matrix = projection_matrix if projection_matrix\
            else mat4.perspective(math.radians(45),float(self.resolution[0])/self.resolution[1],0.1,100)
        pass
    
    def look_at(self, position, target, up):
        self.position = position
        # TODO: decide where to raise the zero vector exception
        # when to check if the it is looking at itself        
        self.orientation = (target - position).normalize()
        self.up = up
        # this look-at method has 3 bugs but still usable by fitting it like this
        self.view_mat = mat4.lookAt(self.position,
                              self.position*2 - target,
                              self.up).inverse()
        return self.view_mat
    
    def on_resize(self):
        pass
    
class Camera_fps(Camera):
    def __init__(self,
                 position=None,
                 spin=0,
                 tilt=0,                 
                 projection_matrix=None,
                 resolution=None):
        # default spin 0 means facing -z, spin positively counter-clockwise
        self.spin = spin
        self.tilt = tilt
        self.position = position if position else vec3(0,0,0)
        self._update_orientation()
        self._update_view_mat()
        self.speed = 1
        self.rev = math.pi / 16
#         Camera.__init__(self,
#                         position,
#                         orientation,
#                         up,
#                         projection_matrix, 
#                         resolution)
        return
    
    def init_input(self, window):
        self.last_mx, self.last_my = glfw.get_cursor_pos(window)
        self.last_time = glfw.get_time()
        self.activate_mouse = False
    
    def _update_orientation(self):
        hori_orientation = vec3(-sin(self.spin),0,-cos(self.spin))
        if abs(self.tilt) != pi/2:
            up = vec3(0,1,0)
            if abs(self.tilt) < pi/4:
                orientation = hori_orientation + vec3(0, tan(self.tilt), 0)
            else:
                tantilt = tan(self.tilt)
                orientation = hori_orientation / tan(self.tilt) + vec3(0,1,0)
                if tantilt < 0: orientation = - orientation 
        else:
            up = -hori_orientation
            orientation = vec3(0,1,0) if self.tilt > 0 else vec3(0,-1,0)
        self.orientation=orientation.normalize()
        self.up = up
        self.right = self.orientation.cross(self.up).normalize()
        self.top = self.right.cross(self.orientation)
        pass
        
    def look_at(self, position, target):
        self.position = position
        self.orientation = (target - position).normalize()
        self.view_mat = mat4.lookAt(self.position,
                              self.position*2 - target,
                              self.up).inverse()
        self.tilt = math.asin(self.orientation.y)
        self.spin = math.atan2(-self.orientation.x, -self.orientation.z)
        self.tilt_max = math.pi * 3 / 8
        return

    def rotate(self, spin, tilt):
        self.spin += spin
        self.tilt += tilt
        self.tilt = max(self.tilt, -self.tilt_max)
        self.tilt = min(self.tilt, self.tilt_max)
        pass
    
    def translate(self, march, sidle, dive):
        self.position += (march * self.orientation +
                          sidle * self.right +
                          dive * self.top)
    
    def _update_view_mat(self):
        res = mat4.translation(-self.position)
        res = mat4.rotation(-self.spin, vec3(0, 1, 0)) * res
        res = mat4.rotation(-self.tilt, vec3(1, 0, 0)) * res
        self.view_mat = res
        return
    
    def bind_input(self, window):
#         glfw.set_key_callback(window, self.keyboard_callback)
#         glfw.set_cursor_pos_callback(window, self.cursor_position_callback)
        pass
    
    def _move_step(self, window, key_p, key_n, del_time):
        if glfw.get_key(window, key_p) == glfw.PRESS:            
            return self.speed * del_time
        elif glfw.get_key(window, key_n) == glfw.PRESS:       
            return - self.speed * del_time
            
    
    def poll_event(self, window):
        flag = False
        cur_time = glfw.get_time()
        del_time = cur_time - self.last_time
        march = sidle = dive = 0.
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:            
            march = self.speed * del_time
        elif glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:            
            march = - self.speed * del_time
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:            
            sidle = self.speed * del_time
        elif glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:            
            sidle = - self.speed * del_time
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:            
            dive = self.speed * del_time
        elif glfw.get_key(window, glfw.KEY_V) == glfw.PRESS:            
            dive = - self.speed * del_time
        if march or sidle or dive:
            self.translate(march, sidle, dive)
            flag = True
        cur_mx, cur_my = glfw.get_cursor_pos(window)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            del_mx, del_my = cur_mx - self.last_mx, cur_my - self.last_my
            spin = tilt = 0.
            if del_mx:
                spin = self.rev * del_mx * del_time
#                 self.spin += self.rev * del_mx
            if del_my:
                tilt = self.rev * del_my * del_time
            if spin or tilt:
                self.rotate(spin, tilt)
                flag = True
        self.last_mx, self.last_my = cur_mx, cur_my
        self.last_time = cur_time
        if flag:
            self._update_orientation()
            self._update_view_mat()
    
    def keyboard_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
#             print "key: %d, scancode: %d, action: %d, mods: %d" % (key, scancode, action, mods)
            if key == glfw.KEY_W:
                self.position += self.speed * self.orientation
            elif key == glfw.KEY_S:
                self.position -= self.speed * self.orientation
            elif key == glfw.KEY_SPACE:
                self.position += self.speed * self.top
            elif key == glfw.KEY_V:
                self.position -= self.speed * self.top
            elif key == glfw.KEY_D:
                self.position += self.speed * self.right
            elif key == glfw.KEY_A:
                self.position -= self.speed * self.right
            elif key == glfw.KEY_LEFT:
                self.spin += self.rev
            elif key == glfw.KEY_RIGHT:
                self.spin -= self.rev
            elif key == glfw.KEY_UP:
                self.tilt = min(self.tilt + self.rev, self.tilt_max)
            elif key == glfw.KEY_DOWN:
                self.tilt = max(self.tilt - self.rev, -self.tilt_max)
            else: 
                return
            self._update_orientation()
            self._update_view_mat()
    
    def cursor_position_callback(self, window, xpos, ypos):
        print xpos, ypos

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
        self.indices = []
        # if the path is not provided, presumably attributes will be filled in
        # afterward
        if path == None:
            return
        else: self.load_from_file(path)
        return
    
    def load_from_file(self, path=None):
        vertices = []
        normals = []
        texcoords = []
        faces = []
#         attrib_map = {'v': vertices,
#                      'vn': normals,
#                      'vt': texcoords}
        append = lambda attrib: lambda line: attrib.append(tuple(map(float, values[1:])))
        ignore = lambda values: None
        def handle_face(values):
            for value in line.split()[1:]:
                # presuming having no negative value
                # converting from 1-base to 0-base
                v = map(lambda x: int(x)-1 if x else None, value.split('/'))
                # presuming drawing triangles only
                faces.append(tuple(v))
        attrib_handle_map = {
                    '#': ignore,    # a comment line
                    '': ignore,     # an empty line
                    'v': append(vertices),
                    'vn': append(normals),
                    'vt': append(texcoords),
                    'f': handle_face
                    }
        # read in from file 
        with open(path, 'r') as obj_file:
            for line in obj_file:
                line = line.strip()                
                if line.startswith('#'): continue # a comment line
                values = line.split()
                if not values: continue # empty line
                try:
                    operation = attrib_handle_map[values[0]]
                    operation(values)
                except KeyError:
                    pass # ignore undefined
#                 # deprecated
#                 if not values: continue # empty line
#                 if values[0] in attrib_map:
#                     attribute = attrib_map[values[0]]
#                     attribute.append(tuple(map(float, values[1:])))
#                 elif values[0] == 'f':
#                     for value in values[1:]:
#                         # presuming having no negative value
#                         # converting from 1-base to 0-base
#                         v = map(lambda x: int(x)-1 if x else None, value.split('/'))
#                         # presuming drawing triangles only
#                         faces.append(tuple(v))
        # indexing to compromise for VBO single index design
        vertex_pack_map = {}
        self.indices = []
        self.vertices = []
        self.normals = []
        self.texcoords = []
        for element in faces:
            try:
                # query for the v-vt-vn indices tuple
                index = vertex_pack_map[element]
            except KeyError:
                # this v-vt-vn pack is not in the dict
                index = len(vertex_pack_map)
                v, vt, vn = element
                vertex_pack_map[element] = index
                for attr_indexed_list, attr_index, attr_list in [
                        (self.vertices, v, vertices),
                        (self.texcoords, vt, texcoords),
                        (self.normals, vn, normals)]:
                    value = attr_list[attr_index] if attr_index != None else None
                    attr_indexed_list.append(value)
#                 if v: self.vertices.append(vertices[v])
#                 if vt: self.texcoords.append(texcoords[vt])
#                 if vn: self.normals.append(normals[vn])
            finally:                
                # by here, element is recorded in the dict
                self.indices.append(index)
        return            
    
    
def _main():
    pass

if __name__ == "__main__":
    print "---starting main()---"
    _main()
    print "---end of main()---"
