import glfw
import tools
from OpenGL.GL import *
from app import *
from ctypes import c_uint8, c_float, c_ushort, c_void_p, c_double
from math import pi, cos, sin
from cgkit.cgtypes import *
from PIL import Image
from threading import Thread, Lock, Semaphore
import numpy as np
import atb
from datetime import datetime as dt
from cgkit.fob import POSITION
from random import random, randint
from cal import *
from _collections import deque
from boto.ec2.autoscale.request import Request
from idlelib.rpc import request_queue

class Renderer(Thread):
    @classmethod
    def shadow_proj_mat(cls, plane_normal, plane_point, light_pos):
        if type(plane_normal) == vec3:
            plane_normal = plane_normal.normalize()
        elif type(plane_normal) == np.array:
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
        else:
            raise TypeError("What the hell did you put in here as a normal??")
        n_t = np.array(plane_normal).reshape((1, 3))
        L = np.array(light_pos).reshape((3, 1))
        D = -plane_normal * plane_point
        ntL = np.dot(n_t, L)
        shad_mat = np.identity(4, float)
        shad_mat[0:3, 0:3] = L.dot(n_t) - (D + ntL) * np.identity(3)
        shad_mat[0:3, 3:4] = (D + ntL) * L - L * ntL
        shad_mat[3:4, 0:3] = n_t
        shad_mat[3:4, 3:4] = -ntL
        return mat4(shad_mat.astype(np.float32).T.tolist())

    class _Item:  # a temporary holder of attributes and uniforms
        color_gen = tools.random_bright_color_generator()
        def __init__(self, model):
            self.model = model
            self.position = vec3(0,0,0)
            self.orientation = vec3(0,0,0)
            self.scale = vec3(1,1,1)
            self.color = vec3(self.color_gen.next())
        
        @staticmethod
        def get_light_color():
            color = vec3(random(), 1, 0)
            i = randint(0, 2)
            color[0], color[i] = color[i], color[0]
            i = randint(0, 1)
            color[0], color[i] = color[i], color[0]
            return color
        
        def model_mat(self):
            # M = "SRT" = T * R * S
            m = mat4(1.0)
            m.translate(self.position)
            rad = self.orientation.length()
            try:
                if rad != 0.0: m.rotate(rad, self.orientation)
            except ZeroDivisionError as e:
#                 print e.message
#                 print rad, self.orientation
                pass
            m.scale(self.scale)
            return m
    
    class _Model: # a holder of obj, buffers and
        def __init__(self, obj):
            self.obj = obj
            self.flatten = lambda l: [u for t in l for u in t]
            
        
#         @staticmethod
#         def flatten(l):
#             return [u for t in l for u in t]
        
        def load_to_buffers(self):
            self.vao_handle = glGenVertexArrays(1)
            glBindVertexArray(self.vao_handle)
            self.i_buffer = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.i_buffer)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                         (c_ushort * len(self.obj.indices))(*self.obj.indices),
                         GL_STATIC_DRAW)
            # vertices buffer
            self.v_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.v_buffer)
            v_flatten = self.flatten(self.obj.vertices)
            glBufferData(GL_ARRAY_BUFFER,
                         (c_float * len(v_flatten))(*v_flatten),
                         GL_STATIC_DRAW)
            # normals buffer
            self.n_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.n_buffer)
            n_flatten = self.flatten(self.obj.normals)
            glBufferData(GL_ARRAY_BUFFER,
                         (c_float * len(n_flatten))(*n_flatten),
                         GL_STATIC_DRAW)
            
        def bind_vao(self):
            glBindVertexArray(self.vao_handle)
#             glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.i_buffer)
#             glBindBuffer(GL_ARRAY_BUFFER, self.v_buffer)
#             glBindBuffer(GL_ARRAY_BUFFER, self.n_buffer)

    class _Window:  # a windows attribute holder
        pass
    
    class _Camera:
        pass
    
    class _Shadow_shader:
        def __init__(self, vert_path, frag_path):
            self.handle = tools.load_program(vert_path, frag_path)
        
        def bind(self, model):
            glBindVertexArray(model.vao_handle)
            self.v_loc = glGetAttribLocation(self.handle, "coord3d")
            glEnableVertexAttribArray(self.v_loc)
            glBindBuffer(GL_ARRAY_BUFFER, model.v_buffer)
            glVertexAttribPointer(self.v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
    
    class _Basic_shader:
        def __init__(self, vert_path, frag_path):
            self.handle = tools.load_program(vert_path, frag_path)
        
        def bind(self, model):
            pass
        
    class _Background_shader:
        def __init__(self, vert_path, frag_path):
            self.handle = tools.load_program(vert_path, frag_path)
            
    class _Shader:
        # a specific wrapping for standard shading
        def __init__(self, vert_path, frag_path):
            self.handle = tools.load_program(vert_path, frag_path)
            
        def bind(self, model):
            glBindVertexArray(model.vao_handle)
            glUseProgram(self.handle)
            self.vert_loc = glGetAttribLocation(self.handle,
                                         "vertexPosition_modelspace")
            glEnableVertexAttribArray(self.vert_loc)
            glBindBuffer(GL_ARRAY_BUFFER, model.v_buffer)
            glVertexAttribPointer(self.vert_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
            # TODO: fix the existing attribute unable to retrieve problem
            self.norm_loc = glGetAttribLocation(self.handle,
                                                     "vertexNormal_modelspace")
            glEnableVertexAttribArray(self.norm_loc)
            glBindBuffer(GL_ARRAY_BUFFER, model.n_buffer)
            glVertexAttribPointer(self.norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

    def set_energy_terms(self, names):
        for name in names:
            if not self._energy_terms.has_key(name):
                self._energy_terms[name] = -23.3
        pass
    
    def set_energy_value(self, name, val):
        try:
            self._energy_terms[name] = val
        except KeyError:
            pass

    def set_penalty_value(self, name, val):
        try:
            self._extern_penalty_terms[name] = val
        except KeyError:
            pass
    
    def set_penalty_terms(self, names):
        for name in names:
            if not self._extern_penalty_terms.has_key(name):
                self._extern_penalty_terms[name] = -23.3
        pass

    def init(self):
        # windows initialization
        self.window = self._Window()        
        self.viewport_size = w, h = (640, 480)
        self.window.width, self.window.height = (w*3/2, h*2)
        self.window.handle = glfw.create_window(self.window.width, self.window.height, "scene", None, None)
        if self.window.handle == None:
            glfw.terminate()
            raise RuntimeError("GLFW cannot create a window.")
        glfw.set_window_pos(self.window.handle, 9, 36)
        glfw.make_context_current(self.window.handle)
        glClearColor(0.0, 0.0, 0.2, 1.0)

        # default blinn-phong shader loading
        self.standard_shader = self._Shader("../shader/standardShading.v.glsl",
                                        "../shader/standardShading.f.glsl")
        self.program_handle = self.standard_shader.handle                                        
        glUseProgram(self.standard_shader.handle)

        self.cube_model = self._Model(Object("../obj/cube.obj"))
        self.sphere_model = self._Model(Object("../obj/sphere/sphere.obj"))
#         self.cube.obj = Object("../obj/cube.obj")
        self.cube = self._Item(self.cube_model)
        self.cube.scale = vec3(0.6, 0.6, 0.6)
        self._items.append(self.cube)
        small_cube = self._Item(self.sphere_model)
        small_cube.scale = vec3(0.2,0.2,0.2)
        small_cube.position = vec3(0.7, 0.1 ,0.7)
        self._items.append(small_cube)
        medium_cube = self._Item(self.cube_model)
        medium_cube.scale = vec3(0.4,0.4,0.4)
        medium_cube.position = vec3(0.2, 0.5, 0.7)
#         medium_cube.position = vec3(-0.7, 0.,0.7)
        self._items.append(medium_cube)
        self.sphere_model.load_to_buffers()
        self.cube_model.load_to_buffers()
        # bind buffers
        # indices buffer
#         self.cube_model.i_buffer = glGenBuffers(1)
#         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cube_model.i_buffer)
#         glBufferData(GL_ELEMENT_ARRAY_BUFFER,
#                      (c_ushort * len(self.cube_model.obj.indices))(*self.cube_model.obj.indices),
#                      GL_STATIC_DRAW)
#         # vertices buffer
#         self.cube_model.v_buffer = glGenBuffers(1)
#         glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.v_buffer)
#         v_flatten = self.flatten(self.cube_model.obj.vertices)
#         glBufferData(GL_ARRAY_BUFFER,
#                      (c_float * len(v_flatten))(*v_flatten),
#                      GL_STATIC_DRAW)
#         # normals buffer
#         self.cube_model.n_buffer = glGenBuffers(1)
#         glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.n_buffer)
#         n_flatten = self.flatten(self.cube_model.obj.normals)
#         glBufferData(GL_ARRAY_BUFFER,
#                      (c_float * len(n_flatten))(*n_flatten),
#                      GL_STATIC_DRAW)

        # attributes initializing
#         self.cube_model.vert_loc = glGetAttribLocation(self.program_handle,
#                                                  "vertexPosition_modelspace")
#         glEnableVertexAttribArray(self.cube_model.vert_loc)
#         glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.v_buffer)
#         glVertexAttribPointer(self.cube_model.vert_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
#         # TODO: fix the existing attribute unable to retrieve problem
#         self.cube_model.norm_loc = glGetAttribLocation(self.program_handle,
#                                                  "vertexNormal_modelspace")
#         glEnableVertexAttribArray(self.cube_model.norm_loc)
#         glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.n_buffer)
#         glVertexAttribPointer(self.cube_model.norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        # cube uniforms
#         self.cube.position = vec3(0.5, 0, 1)
#         self.cube.model_mat = mat4(1.0)
#         self.cube.model_mat.translate(self.cube.position)
#         self.cube.model_mat.rotate(pi / 3, vec3(1.0, 0.5, 1.7))
#         self.cube.model_mat.scale(vec3(0.5))

        # camera initializing
        self.cam_obs = self._Camera()  # the camera for human observation
        self.cam_obs.view_mat = self.look_at(vec3(-1, 2, 5),
                                   vec3(0, 0, 0),
                                   vec3(0, 1, 0))
        self.cam_cap = self._Camera()  # the camera to capture shadow
        self.cam_cap.view_mat = self.look_at(vec3(0, 4, 0),
                                   vec3(0, 0, 0),
                                   vec3(0, 0, -1))
        self.cam_cap.proj_mat = self.cam_obs.proj_mat = mat4.perspective(45, 4. / 3, 0.1, 100)
#         self.model_view_inv = (self.cam_obs.view_mat * item.model_mat).inverse()
    #     light_pos = vec3(2,1,0)
    #     light_pos = vec3(2,2,2)
        self.light_pos = vec3(3, 3, 3)
        self.cam_obs.V_loc = glGetUniformLocation(self.program_handle, "V")
        glUniformMatrix4fv(self.cam_obs.V_loc, 1, GL_FALSE, self.cam_obs.view_mat.toList())
        self.light_pos_loc = glGetUniformLocation(self.program_handle,
                                                  "LightPosition_worldspace")
        glUniform3f(self.light_pos_loc,
                    self.light_pos.x, self.light_pos.y, self.light_pos.z)
        self.MVP_loc = glGetUniformLocation(self.program_handle, "MVP")
#         self.cube.MVP = mat4(1)
        self.M_loc = glGetUniformLocation(self.program_handle, "M")
        self.MVint_loc = glGetUniformLocation(self.program_handle, "MVint")
        self.color_loc = glGetUniformLocation(self.standard_shader.handle, "MaterialDiffuseColor")

        # init the floor
        self.floor = self._Item(self.cube_model)
        self.floor.position = vec3((0, -0.51, 0))
        self.floor.scale = vec3(5, 0.1, 5)
#         self.floor.model_mat = mat4.translation((0, -0.51, 0)) * mat4.scaling((5, 0.1, 5))
#         self.floor.MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * self.floor.model_mat
#         self.floor.MVinv = (self.cam_obs.view_mat * self.floor.model_mat).inverse()

        # initialize shadow projection program
        self.shadow = self._Shadow_shader("../shader/shadowProjectionShading.v.glsl",
                                    "../shader/shadowProjectionShading.f.glsl")
        self.shadow_program_handle = self.shadow.handle
        glUseProgram(self.shadow_program_handle)
        self.shadow.MsVP_loc = glGetUniformLocation(self.shadow_program_handle, "MsVP")
        self.shadow.VP_mat = self.cam_obs.proj_mat * self.cam_obs.view_mat;
        self.shadow.VP_mat_top = self.cam_cap.proj_mat * self.cam_cap.view_mat;
        self.shadow.shaject_mat = self.shadow_proj_mat(vec3(0, 1, 0), vec3(0, -0.45, 0), self.light_pos)
        self.shadow.color_loc = glGetUniformLocation(self.shadow.handle, "shadowColor")
        glUniform3f(self.shadow.color_loc, 0.0, 0.0, 0.0)  # black shadow
        self.shadow.alpha_loc = glGetUniformLocation(self.shadow.handle, "alpha")

        # init the shader to draw the basic shadow
        self.basic_program_handle = tools.load_program("../shader/basic.v.glsl",
                                                   "../shader/basic.f.glsl")
        glUseProgram(self.basic_program_handle)
        self.basic_mvp_loc = glGetUniformLocation(self.basic_program_handle, "mvp")
        basic_mvp = self.cam_cap.proj_mat * self.cam_cap.view_mat * self.floor.model_mat()
        glUniformMatrix4fv(self.basic_mvp_loc, 1, GL_FALSE, basic_mvp.toList())
        self.basic_v_loc = glGetAttribLocation(self.basic_program_handle, "coord3d")
        glEnableVertexAttribArray(self.basic_v_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.v_buffer)
        glVertexAttribPointer(self.basic_v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        self.background_indices = range(4) # will use triangle fans
        far_end = 1 - 1e-4
        self.background_vert = [-1.,  1.,  far_end,
                                -1., -1.,  far_end,
                                 1., -1.,  far_end,
                                 1.,  1.,  far_end] # in clip coordinates
        self.background_uvs = [0., 1.,
                               0., 0.,
                               1., 0.,
                               1., 1.]
        self.bg_shader = self._Background_shader("../shader/background.v.glsl", 
                                                 "../shader/background.f.glsl")
        self.background_vao_handle = glGenVertexArrays(1)
        glBindVertexArray(self.background_vao_handle)
        self.bg_i_buffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.bg_i_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     (c_ushort * len(self.background_indices))(*self.background_indices),
                     GL_STATIC_DRAW)
        self.bg_v_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.bg_v_buffer)
        glBufferData(GL_ARRAY_BUFFER,
                     (c_float * len(self.background_vert))(*self.background_vert),
                     GL_STATIC_DRAW)
        # uv buffer
        self.bg_uv_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.bg_uv_buffer)
        glBufferData(GL_ARRAY_BUFFER,
                     (c_float * len(self.background_uvs))(*self.background_uvs),
                     GL_STATIC_DRAW)
        self.bg_tex_handle = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex_handle)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        try:
            im = self.bg_img
            ix, iy, image = im.size[0], im.size[1], im.tostring("raw", "L", 0, -1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ix, iy, 0,
                         GL_RED, GL_UNSIGNED_BYTE, image)
        except AttributeError: # target image as comparison background is not set
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 1, 1, 0,
                         GL_RED, GL_UNSIGNED_BYTE, '\xff')
        
#             raise RuntimeError()
        # initializing other stuff
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        
        
#         _init_X_Y(self.window.width, self.window.height)

#         self._X = np.arange(self.window.width).reshape(1,self.window.width)
#         self._Y = np.arange(self.window.height).reshape(self.window.height,1)
        self.init_atb()
        self._init_target_image
    
    def init_atb(self):
#         atb.init() # cannot tell what for, given by the binding author
        self.total = -233.3
        atb.TwInit(atb.TW_OPENGL, None)
        atb.TwWindowSize(self.window.width, self.window.height)
        self.extern_param_bar = atb.Bar(name="extern_param", label="evals", help="Scene atb",
                           position=(650, 10), size=(300,300), valueswidth=150)
        self.extern_param_bar.add_var("total", getter=self.get_total)
        # external defined energy terms
        self.extern_param_bar.add_separator("")
        for i, name in enumerate(self._energy_terms):
            self.extern_param_bar.add_var(name="energy_term_%d" % i, label=name, 
                    getter=self._disp_getter_closure(self._energy_terms, name))
        
        # external defined penalties
        self.extern_param_bar.add_separator("")
        for i, name in enumerate(self._extern_penalty_terms):
            self.extern_param_bar.add_var(name="extern_penalty_%d" % i, label=name,
                    getter=self._disp_getter_closure(self._extern_penalty_terms, name))
        
        # internal penalties
        self.extern_param_bar.add_separator("")
        for i, name in enumerate(self._intern_penalty_terms):
            self.extern_param_bar.add_var(name="intern_penalty_%d" % i, label=name,
                    getter=self._disp_getter_closure(self._intern_penalty_terms, name))
        
        self.control_bar = atb.Bar(name="controls", label="controls",
                                   position=(650, 320), size=(300, 300), valueswidth=150)
        def position_getter_closure(item, index):
            def getter():
                return item.position[index]
            return getter   
        def position_setter_closure(item, index):
            def setter(x):
                item.position[index] = x
            return setter        
        def rotation_getter_closure(item, index):
            def getter():
                return item.orientation[index]
            return getter   
        def rotation_setter_closure(item, index):
            def setter(x):
                item.orientation[index] = x
            return setter
        for i, item in enumerate(self._items):
            group = "item_%d" % i
            for j, n in enumerate('xyz'):
                name = "%s %s" % (group, n)
                self.control_bar.add_var(name=name, label=n, readonly=False, 
                                         vtype=atb.TW_TYPE_FLOAT, step=0.05,
                                         group=group,
                                         getter=position_getter_closure(item, j), 
                                         setter=position_setter_closure(item, j))
            for j, n in enumerate('abc'):
                name = "%s %s" % (group, n)
                self.control_bar.add_var(name=name, label=n, readonly=False, 
                                         vtype=atb.TW_TYPE_FLOAT, step=0.05,
                                         group=group,
                                         getter=rotation_getter_closure(item, j), 
                                         setter=rotation_setter_closure(item, j))
            self.control_bar.define("opened=false",group)
        
        atb.TwDefine("extern_param refresh=0.1")
        def mouse_button_callback(window, button, action, mods):
            tAction = tButton = -1
            if action == glfw.RELEASE:
                tAction = atb.TW_MOUSE_RELEASED
            elif action == glfw.PRESS:
                tAction = atb.TW_MOUSE_PRESSED
            if button == glfw.MOUSE_BUTTON_LEFT:
                tButton = atb.TW_MOUSE_LEFT
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                tButton = atb.TW_MOUSE_RIGHT
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                tButton = atb.TW_MOUSE_MIDDLE
            if not (tAction == -1 or tButton == -1): atb.TwMouseButton(tAction, tButton)            
        glfw.set_mouse_button_callback(self.window.handle, mouse_button_callback)
        cursor_callback = lambda w, x, y: atb.TwMouseMotion(int(x),int(y))
        glfw.set_cursor_pos_callback(self.window.handle, cursor_callback)
        pass
    
    def get_total(self):
        return self.total
    
    def set_total(self, total):
        self.total = total
    
    def _disp_getter_closure(self, dict, name):
        def getter():
            return dict[name]
        return getter

    def get_mat_model2snapshot(self):
        mat_shaj = self.shadow.shaject_mat
        mat_view = self.cam_cap.view_mat
        mat_proj = self.cam_cap.proj_mat
        w, h = self.viewport_size
        # mat_c2s is projection from clip coordinate to screen coordinate
        mat_c2s = mat4([[(w-1)/2, 0, 0, (w-1)/2],
                        [0, (1-h)/2, 0, (h-1)/2],
                        [0,       0, 0,       0],
                        [0,       0, 0,       1]])
        mat_c2s = mat_c2s.transpose() # cgkit.mat4 initialization is column major
        mat_c2s = mat_c2s * mat_proj * mat_view * mat_shaj
        return mat_c2s
    
    def model_center_penalty_closure(self, img):
        # presuming the camera for capturing, light and receiver won't move
        # or otherwise all the mat shall be computed on the fly
        mat_c2s = self.get_mat_model2snapshot()
        fst_moment = get_fst_moments(img)
        def model_center_penalty(x):
            position = self._items[0].position
            position = mat_c2s*position
            self._intern_penalty_terms['x'] = position.x - fst_moment[0]
            self._intern_penalty_terms['y'] = position.y - fst_moment[1]
            return 0
        return model_center_penalty
    
    def set_target_image(self, img):
        if img.getbands() != ('L',):
            img = img.convert('L')
        self.image_target = img
        self.bg_img = img
        # maybe some other preprocessing
    
    def _init_target_image(self):
        self.scene_penalty = self.model_center_penalty_closure(self.image_target)

    def scene_penalty(self):
        # simply a place holder, will be replaced by setting image for
        return 0 
        
    def __init__(self):
        Thread.__init__(self)
        if not glfw.init():
            raise RuntimeError("Cannot start up GLFW")
        self.flatten = lambda l: [u for t in l for u in t]
        self.c_array = lambda c_type: lambda l: (c_type * len(l))(*l)
        # TODO: rewrite this stupid expedient look_at function
        self.look_at = lambda eye, at, up: mat4.lookAt(eye, 2 * eye - at, up).inverse()
#         self.init()
        self.ss_update = Lock() # ss is short for snapshot
#         self.ss_ready = Lock()
#         self.ss_ready.acquire()
        self.ss_ready = Semaphore(0)
        self.snapshot = None
        self.param_lock = Lock()
        self._init_finished_lock = Lock()
        self._init_finished_lock.acquire()
        self._finalized_lock = Lock()
        self._finalized_lock.acquire()
        self._items = []
        self._cont_flag = True
        self._energy_terms = {}
        self._extern_penalty_terms = {}
        self._intern_penalty_terms = {"shadow_distance": -23.3, 'x': -23.3, 'y': -23.3}
        self.bg_img = None
    
    def stop(self):
        self._cont_flag = False

    def draw(self, x):
        
#         self.set_param(x)
        # Render here
        # Make the window's context current
        self.shadow.shaject_mat = self.shadow_proj_mat(vec3(0, 1, 0), 
                                                       vec3(0, -0.45, 0), 
                                                       self.light_pos)

        glfw.make_context_current(self.window.handle)
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        w, h = self.viewport_size
        # draw the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE)
        glViewport(0, 0, w, h)
        glUseProgram(self.program_handle)
        glUniform3f(self.light_pos_loc, *self.light_pos)
        for item in self._items:
            self.standard_shader.bind(item.model)
            model_view_inv = (self.cam_obs.view_mat * item.model_mat()).inverse()
            glUniformMatrix4fv(self.MVint_loc, 1, GL_TRUE, model_view_inv.toList())
            glUniformMatrix4fv(self.M_loc, 1, GL_FALSE, item.model_mat().toList())
            
            MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * item.model_mat()
            glUniformMatrix4fv(self.MVP_loc, 1, GL_FALSE, MVP.toList())
            glUniform3f(self.color_loc, *item.color)
            glDrawElements(GL_TRIANGLES, len(item.model.obj.indices),
                            GL_UNSIGNED_SHORT, None);
        MVint = (self.cam_obs.view_mat * self.floor.model_mat()).inverse()
        glUniformMatrix4fv(self.MVint_loc, 1, GL_TRUE, MVint.toList())
        M = self.floor.model_mat()
        glUniformMatrix4fv(self.M_loc, 1, GL_FALSE, M.toList())
        MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * M
        glUniformMatrix4fv(self.MVP_loc, 1, GL_FALSE, MVP.toList())
        glUniform3f(self.color_loc, 1., 1., 1.)
        glDrawElements(GL_TRIANGLES, len(self.floor.model.obj.indices),
                        GL_UNSIGNED_SHORT, None)
#         glDisable(GL_CULL_FACE)
        # draw the projected shadows8
        glUseProgram(self.shadow_program_handle)
        glUniform3f(self.shadow.color_loc, 0.0, 0.0, 0.0)  # black shadow
#         glUniform1f(self.shadow.alpha_loc, 0.5)
        for item in self._items:
            self.shadow.bind(item.model)
            glUniformMatrix4fv(self.shadow.MsVP_loc, 1, GL_FALSE,
                   (self.shadow.VP_mat * self.shadow.shaject_mat * item.model_mat()).toList())
            glDrawElements(GL_TRIANGLES, len(item.model.obj.indices),
                            GL_UNSIGNED_SHORT, None)

        glViewport(0, h, w, h)
#         glDisable(GL_CULL_FACE)
        
#         glUseProgram(self.basic_program_handle)
#         # needs binding the floor
#         glDrawElements(GL_TRIANGLES, len(self.floor.model.obj.indices),
#                         GL_UNSIGNED_SHORT, None)
        
        glUseProgram(self.bg_shader.handle)
        glBindVertexArray(self.background_vao_handle)
#         glBindVertexArray(self.floor.model.vao_handle)
        v_loc = glGetAttribLocation(self.bg_shader.handle, "coord3d")
        glEnableVertexAttribArray(v_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.bg_v_buffer)
#         glBindBuffer(GL_ARRAY_BUFFER, self.floor.model.v_buffer)
        glVertexAttribPointer(v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
#         uv_loc = glGetAttribLocation(self.bg_shader.handle, "vertexUV")
        uv_loc = 2
        glEnableVertexAttribArray(uv_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.bg_uv_buffer)
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
        bg_tex_loc = glGetUniformLocation(self.bg_shader.handle, "bg_tex")
        glUniform1i(bg_tex_loc, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex_handle)
        glDrawElements(GL_TRIANGLE_FAN, len(self.background_indices), GL_UNSIGNED_SHORT, None)


#         glPushMatrix()  
#         glLoadIdentity()
#         glBegin(GL_TRIANGLES)
#         glColor3f(1., 0., 0.)
#         glVertex3f(-1., 1., 0.)
#         glVertex3f(1., -1., 0.)
#         glVertex3f(1., 1., 0.)
#         glEnd()
#         glPopMatrix()
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(self.shadow_program_handle)
        glEnable(GL_CULL_FACE)
        glUniform3f(glGetUniformLocation(self.shadow.handle, "shadowColor"),
                    0.0, 1.0, 1.0)  # black shadow
        glUniform1f(glGetUniformLocation(self.shadow.handle, "alpha"),
                    0.75)
        for item in self._items:
            self.shadow.bind(item.model)
            glUniformMatrix4fv(self.shadow.MsVP_loc, 1, GL_FALSE,
                   (self.shadow.VP_mat_top * self.shadow.shaject_mat * item.model_mat()).toList())
    
    #         glUniformMatrix4fv(shadow_M_loc, 1, GL_FALSE, model_mat.toList())
    #         glUniformMatrix4fv(shadow_VP_loc, 1, GL_FALSE, VP_mat_top.toList())
            glDrawElements(GL_TRIANGLES, len(item.model.obj.indices),
                            GL_UNSIGNED_SHORT, None)
        glDisable(GL_BLEND)
        
        glViewport(w, 0, w/2, h/2)
        glUseProgram(self.bg_shader.handle)
        glBindVertexArray(self.background_vao_handle)
#         glBindVertexArray(self.floor.model.vao_handle)
        v_loc = glGetAttribLocation(self.bg_shader.handle, "coord3d")
        glEnableVertexAttribArray(v_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.bg_v_buffer)
#         glBindBuffer(GL_ARRAY_BUFFER, self.floor.model.v_buffer)
        glVertexAttribPointer(v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
#         uv_loc = glGetAttribLocation(self.bg_shader.handle, "vertexUV")
        uv_loc = 2
        glEnableVertexAttribArray(uv_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.bg_uv_buffer)
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
        bg_tex_loc = glGetUniformLocation(self.bg_shader.handle, "bg_tex")
        glUniform1i(bg_tex_loc, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex_handle)
        glDrawElements(GL_TRIANGLE_FAN, len(self.background_indices), GL_UNSIGNED_SHORT, None)
#         glUseProgram(self.basic_program_handle)
#         glDrawElements(GL_TRIANGLES, len(self.floor.model.obj.indices),
#                         GL_UNSIGNED_SHORT, None)
        
        glUseProgram(self.shadow_program_handle)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUniform1f(glGetUniformLocation(self.shadow.handle, "alpha"), 0.5)
        glUniform1i(glGetUniformLocation(self.shadow.handle, "change_depth"), 1)
        for item in self._items:
            self.shadow.bind(item.model)
            glUniformMatrix4fv(self.shadow.MsVP_loc, 1, GL_FALSE,
                   (self.shadow.VP_mat_top * self.shadow.shaject_mat * item.model_mat()).toList())
#             glUniform3f(glGetUniformLocation(self.shadow.handle, "shadowColor"),
#                         0., 0., 0.)
#             glDrawElements(GL_LINE_LOOP, len(item.model.obj.indices),
#                             GL_UNSIGNED_SHORT, None)
            glUniform3f(glGetUniformLocation(self.shadow.handle, "shadowColor"),
                        *item.color)
            glUniform1f(glGetUniformLocation(self.shadow.handle, "clip_depth"),
                        1/(item.position.y**2 + 1.001))
    #         glUniformMatrix4fv(shadow_M_loc, 1, GL_FALSE, model_mat.toList())
    #         glUniformMatrix4fv(shadow_VP_loc, 1, GL_FALSE, VP_mat_top.toList())
            glDrawElements(GL_TRIANGLES, len(item.model.obj.indices),
                            GL_UNSIGNED_SHORT, None)
        glUniform1i(glGetUniformLocation(self.shadow.handle, "change_depth"), 0)
        glDisable(GL_BLEND)
        
        atb.TwDraw()
            
        # handling snapshot request if any
        if self.ss_update.locked():
            self._save_snapshot()
            self.ss_ready.release()
#             self.ss_update.release()
        # Swap front and back buffers
#         self._save_snapshot()


        glfw.swap_buffers(self.window.handle)
        glfw.poll_events()

    # called by external threads
    def acquire_snapshot(self):
        self.ss_update.acquire()
        self.ss_ready.acquire()
        self.ss_update.release()
#         self.param_lock.acquire()
        band_index = 0 # using red band to store a shadow
        w, h = self.viewport_size
        img = self.snapshot.crop((0, h, w, h * 2))
        img.load()
        img = img.split()[0]
        if max(img.getdata()) == 0:
            raise RuntimeError("All black encountered")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
    
    def acquire_full_snapshot(self):
        self.ss_update.acquire()
        self.ss_ready.acquire()
        self.ss_update.release()
#         self.param_lock.acquire()
        img = self.snapshot.copy()
        if max(img.getdata()) == 0:
            raise RuntimeError("All black encountered")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
        
    def _save_snapshot(self):
#         glfw.swap_buffers(self.window.handle)
        prev_buff_read = glGetIntegerv(GL_READ_BUFFER)
        glReadBuffer(GL_BACK)
        glFinish()
        buff = glReadPixels(0, 0, self.window.width, 
                         self.window.height, GL_RGB, GL_UNSIGNED_BYTE)
        glReadBuffer(prev_buff_read)
#         glfw.swap_buffers(self.window.handle)
        img = Image.fromstring(mode="RGB", data=buff, 
                              size=(self.window.width, self.window.height))
#         self.param_lock.release()
        self.snapshot = img
    
    def set_param(self, x):
        self.param_lock.acquire()
        for i, item in enumerate(self._items):
            i *= 6
            item.position = vec3(x[i: i + 3])
            item.orientation = vec3(x[i + 3: i + 6])
#         big_cube = self._items[0]
#         big_cube.position = vec3(x[0:3])
#         big_cube.orientation = vec3(x[3:6])
        
        self.param_lock.release()
        return
    
    def set_param_indiv(self, x_i, i):
        # set one individual component of the parameters
        x = self.get_param()
        x[i] = x_i
        self.set_param(x)
    
    def get_param(self):
#         big_cube = self._items[0]
        params = []
        for item in self._items:
            params.append(item.position)
            params.append(item.orientation)
        return np.concatenate(params)

    def to_close():
        return False
    
    def wait_till_init(self):
        if self._init_finished_lock.locked(): # init is ongoing 
            self._init_finished_lock.acquire()
            self._init_finished_lock.release()
        return
    
    def wait_till_final(self):
        if self._finalized_lock.locked():
            self._finalized_lock.acquire()
            self._finalized_lock.release()
        return

    def run(self):
#         draw_projected_shadows()
        
        self.init()
        self._init_finished_lock.release()
#         self.optimize()
        while not glfw.window_should_close(self.window.handle) and self._cont_flag:
            self.param_lock.acquire()
            glDrawBuffer(GL_BACK)
            self.draw(None)
            self.param_lock.release()
            err = glGetError()
            if err != GL_NO_ERROR: print "Encountered a glError:", err
        atb.shutdown()
        glfw.terminate()
        self._finalized_lock.release()
        pass


from threading import Semaphore, Lock
from datetime import datetime as dt
class Request_deque():
    from collections import deque
    def __init__(self, value=1):
        self.sema = Semaphore(value)
        self.time_stamp_q = deque()
        self.sync_lock = Lock()
    
    def acquire(self, blocking=True):
        if self.sema.acquire(blocking):
            # released under blocked mode or happened to have spare under
            #non-blocking mode
            return True, self.time_stamp_q.popleft()
        else:                    
            # non-blocking mode with unsuccessful acquiring
            return False, None
    
    def release(self, stop=False):
        with self.sync_lock:
            # need to guarantee the order matching between request and time
            #stamp, the operation shall be atomic. This could be rare to have
            #but unaffordable if any.
            if stop:
                self.time_stamp_q.append(None)
            else:
                self.time_stamp_q.append(dt.now())
            self.sema.release()
            
    ### END OF Request_deque


class Renderer_dispatcher(Thread):
    '''
    the only interface to acquire rendering thread in the future, as renderer
    factory. Shall be a singleton. Provide unified management
    of OpenGL context, i.e. initializing and deconstructing renderer.
    '''
    def __init__(self):
        Thread.__init__(self)
        self.callbacks = {}
        self.renderer = None
        self.requests = Request_deque(0)
        self.response = Semaphore(0)
        self.last_instance_time = None
        self.stop_flag = False
        self.booting_lock = Lock()
        self.callback_lock = Lock()
        self.energy_terms = None
        self.penalty_terms = None
        self.target_image = None
        pass
    
    def stop(self):
        self.stop_flag = True
        self.renderer.stop()
        self.renderer.wait_till_final()
        self.requests.release(stop=True) # in case dispatcher is blocked.
    
    def run(self):
        while not self.stop_flag: # in case block in handling request, a false request will be pushed
            # handling request
            status, timest = self.requests.acquire()
            if status and not timest:
                # false request encountered
                break
            # reboot a renderer
            if not self.last_instance_time\
                    or timest > self.last_instance_time:
                # only reboot when the last instance is created far ago
                # in-another way: don't reboot if we just did - multiple duplicate
                # request could come in in a a short time if the shared renderer
                # screw up
                self._reboot()
            self.response.release()
        return
    
    def acquire_new(self, energy_terms=None, penalty_terms=None, target_image=None):
        # note that the parameters will be overwritten by the later request if the
        #rebooting process is not finished, ending at having one with different parameter
        #than requested. This is terrible but I cannot figure any better way.
        if energy_terms: self.energy_terms = energy_terms
        if penalty_terms: self.penalty_terms = penalty_terms
        if target_image: self.target_image = target_image
        self.requests.release() # posting a reboot request
        self.response.acquire() # wait until the request has been responded
        return self.renderer # at this point renderer shall have been rebooted
    
    def acquire(self):
        # acquire the current one without booting unless it is None, blocked if under booting process
        if self.booting_lock.locked():
            with self.booting_lock:
                return self.renderer
        elif self.renderer:
            return self.renderer
        else:
            return self.acquire_new()
    
    def register(self, callback, *args, **kwds):
        '''
        callback will be invoke when the renderer is rebooted.
        re-registering same callback with different argument is not supported,
        the former argument will be overwritten and the callback will be called
        only once
        '''
        with self.callback_lock:
            self.callbacks[callback] = (args, kwds)
    
    def deregister(self, callback):
        try:
            del self.callbacks[callback]
        except KeyError:
            pass # doesn't matter
    
    def _reboot(self):
        # shut down the previous one if any
        if self.renderer:
            self.renderer.stop()
            self.renderer.wait_till_final()
        # boot the renderer
        self.renderer = Renderer()
        if self.energy_terms:
            self.renderer.set_energy_terms(self.energy_terms)
        if self.penalty_terms:
            self.renderer.set_penalty_terms(self.penalty_terms)
        if self.target_image:
            self.renderer.set_target_image(self.target_image)
        self.renderer.start()
        self.renderer.wait_till_init()
        # -- need to collect the parameters
        self.last_instance_time = dt.now()
        # call all registered callbacks
        with self.callback_lock:
            for callback, param in self.callbacks.iteritems():
                args, kwds = param
                callback(*args, **kwds)
        pass
    
    ### END OF Renderer_dispatcher

    

def _main():
    renderer = Renderer()   
    renderer.set_energy_terms(['energy_1', 'energy_2', 'energy_3'])
    renderer.set_penalty_terms(['energy_1', 'energy_2', 'energy_3'])
    im = Image.open("..\\img\\target_mickey.png")
    im = im.convert('L')
    renderer.set_target_image(im)
#     renderer.bg_img = im
    renderer.start()
    renderer.wait_till_init()
#     renderer.scene_penalty(1)
#     from shadow_optim import MyGUI
#     gui = MyGUI(renderer)
#     gui.run()
    return

if __name__ == "__main__":
    print "----------start of main---------"
    _main()
    print "---------end of main-------------"
