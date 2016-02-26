import glfw
import tools
from OpenGL.GL import *
from app import *
from ctypes import c_uint8, c_float, c_ushort, c_void_p
from math import pi, cos, sin
from cgkit.cgtypes import *
from PIL import Image
from threading import Thread, Lock, Semaphore

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
        def __init__(self, model):
            self.model = model
            self.position = vec3(0,0,0)
            self.orientation = vec3(0,0,0)
            self.scale = vec3(1,1,1)
        
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
        pass

    class _Window:  # a windows attribute holder
        pass
    
    class _Camera:
        pass
    
    class _Shadow:
        pass

    def init(self):
        # windows initialization
        self.window = self._Window()
        self.window.width, self.window.height = (640, 480)
        self.window.handle = glfw.create_window(self.window.width * 2, self.window.height, "scene", None, None)
        if self.window.handle == None:
            glfw.terminate()
            raise RuntimeError("GLFW cannot create a window.")
        glfw.set_window_pos(self.window.handle, 9, 36)
        glfw.make_context_current(self.window.handle)
        glClearColor(0.0, 0.0, 0.2, 1.0)

        # default blinn-phong shader loading
        self.program_handle = tools.load_program("../shader/standardShading.v.glsl",
                                        "../shader/standardShading.f.glsl")
        glUseProgram(self.program_handle)

        self.cube_model = self._Model(Object("../obj/cube.obj"))
#         self.cube.obj = Object("../obj/cube.obj")
        self.cube = self._Item(self.cube_model)
        self.cube.scale = vec3(0.6, 0.6, 0.6)
        self._items.append(self.cube)
        small_cube = self._Item(self.cube_model)
        small_cube.scale = vec3(0.2,0.2,0.2)
        small_cube.position = vec3(0.7, 0.,0.7)
        self._items.append(small_cube)
        medium_cube = self._Item(self.cube_model)
        medium_cube.scale = vec3(0.4,0.4,0.4)
        medium_cube.position = vec3(-0.7, 0.,0.7)
        self._items.append(medium_cube)
        # initialize VAO
        self.vao_handle = glGenVertexArrays(1)
        glBindVertexArray(self.vao_handle)

        # bind buffers
        # indices buffer
        self.cube_model.i_buffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cube_model.i_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     (c_ushort * len(self.cube_model.obj.indices))(*self.cube_model.obj.indices),
                     GL_STATIC_DRAW)
        # vertices buffer
        self.cube_model.v_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.v_buffer)
        v_flatten = self.flatten(self.cube_model.obj.vertices)
        glBufferData(GL_ARRAY_BUFFER,
                     (c_float * len(v_flatten))(*v_flatten),
                     GL_STATIC_DRAW)
        # normals buffer
        self.cube_model.n_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.n_buffer)
        n_flatten = self.flatten(self.cube_model.obj.normals)
        glBufferData(GL_ARRAY_BUFFER,
                     (c_float * len(n_flatten))(*n_flatten),
                     GL_STATIC_DRAW)

        # attributes initializing
        self.cube_model.vert_loc = glGetAttribLocation(self.program_handle,
                                                 "vertexPosition_modelspace")
        glEnableVertexAttribArray(self.cube_model.vert_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.v_buffer)
        glVertexAttribPointer(self.cube_model.vert_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
        # TODO: fix the existing attribute unable to retrieve problem
        self.cube_model.norm_loc = glGetAttribLocation(self.program_handle,
                                                 "vertexNormal_modelspace")
        glEnableVertexAttribArray(self.cube_model.norm_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.n_buffer)
        glVertexAttribPointer(self.cube_model.norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

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

        # init the floor
        self.floor = self._Item(self.cube_model)
        self.floor.position = vec3((0, -0.51, 0))
        self.floor.scale = vec3(5, 0.1, 5)
#         self.floor.model_mat = mat4.translation((0, -0.51, 0)) * mat4.scaling((5, 0.1, 5))
#         self.floor.MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * self.floor.model_mat
#         self.floor.MVinv = (self.cam_obs.view_mat * self.floor.model_mat).inverse()

        # initialize shadow projection program
        self.shadow_program_handle = tools.load_program("../shader/shadowProjectionShading.v.glsl",
                                                   "../shader/shadowProjectionShading.f.glsl")
        glUseProgram(self.shadow_program_handle)
        self.shadow = self._Shadow()
        self.shadow.MsVP_loc = glGetUniformLocation(self.shadow_program_handle, "MsVP")
        self.shadow.VP_mat = self.cam_obs.proj_mat * self.cam_obs.view_mat;
        self.shadow.VP_mat_top = self.cam_cap.proj_mat * self.cam_cap.view_mat;
        self.shadow.shaject_mat = self.shadow_proj_mat(vec3(0, 1, 0), vec3(0, -0.45, 0), self.light_pos)
        glUniform3f(glGetUniformLocation(self.shadow_program_handle, "shadowColor"),
                     0.0, 0.0, 0.0)  # black shadow
        self.shadow.v_loc = glGetAttribLocation(self.shadow_program_handle, "coord3d")
        glEnableVertexAttribArray(self.shadow.v_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_model.v_buffer)
        glVertexAttribPointer(self.shadow.v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

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

        # initializing other stuff
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        
#         _init_X_Y(self.window.width, self.window.height)

#         self._X = np.arange(self.window.width).reshape(1,self.window.width)
#         self._Y = np.arange(self.window.height).reshape(self.window.height,1)
    
    # obsolete
    def set_target_image(self, filepath):
        self.image_target = Image.open(filepath)
        self.image_target = self.image_target.convert("L")
        # TODO: process image if unmatch with

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
        self._items = []

    def draw(self, x):
#         self.set_param(x)
        # Render here
        # Make the window's context current
        self.shadow.shaject_mat = self.shadow_proj_mat(vec3(0, 1, 0), 
                                                       vec3(0, -0.45, 0), 
                                                       self.light_pos)

        glfw.make_context_current(self.window.handle)
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE)
        glViewport(0, 0, self.window.width, self.window.height)
        glUseProgram(self.program_handle)
        glUniform3f(self.light_pos_loc, *self.light_pos)
        for item in self._items:
            model_view_inv = (self.cam_obs.view_mat * item.model_mat()).inverse()
            glUniformMatrix4fv(self.MVint_loc, 1, GL_TRUE, model_view_inv.toList())
            glUniformMatrix4fv(self.M_loc, 1, GL_FALSE, item.model_mat().toList())
            MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * item.model_mat()
            glUniformMatrix4fv(self.MVP_loc, 1, GL_FALSE, MVP.toList())
            glDrawElements(GL_TRIANGLES, len(item.model.obj.indices),
                            GL_UNSIGNED_SHORT, None);
        MVint = (self.cam_obs.view_mat * self.floor.model_mat()).inverse()
        glUniformMatrix4fv(self.MVint_loc, 1, GL_TRUE, MVint.toList())
        M = self.floor.model_mat()
        glUniformMatrix4fv(self.M_loc, 1, GL_FALSE, M.toList())
        MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * M
        glUniformMatrix4fv(self.MVP_loc, 1, GL_FALSE, MVP.toList())
        glDrawElements(GL_TRIANGLES, len(self.floor.model.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        glDisable(GL_CULL_FACE)
        # draw the projected shadows
        glUseProgram(self.shadow_program_handle)
        for item in self._items:
            glUniformMatrix4fv(self.shadow.MsVP_loc, 1, GL_FALSE,
                   (self.shadow.VP_mat * self.shadow.shaject_mat * item.model_mat()).toList())
            glDrawElements(GL_TRIANGLES, len(item.model.obj.indices),
                            GL_UNSIGNED_SHORT, None)


        glViewport(self.window.width, 0, self.window.width, self.window.height)

        glDisable(GL_CULL_FACE)
        glUseProgram(self.basic_program_handle)
        glDrawElements(GL_TRIANGLES, len(self.floor.model.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        
        glUseProgram(self.shadow_program_handle)
        for item in self._items:
            glUniformMatrix4fv(self.shadow.MsVP_loc, 1, GL_FALSE,
                   (self.shadow.VP_mat_top * self.shadow.shaject_mat * item.model_mat()).toList())
    
    #         glUniformMatrix4fv(shadow_M_loc, 1, GL_FALSE, model_mat.toList())
    #         glUniformMatrix4fv(shadow_VP_loc, 1, GL_FALSE, VP_mat_top.toList())
            glDrawElements(GL_TRIANGLES, len(item.model.obj.indices),
                            GL_UNSIGNED_SHORT, None)
#         if self.ss_update.locked():
#             self._save_snapshot()
#             self.ss_ready.release()
#             self.ss_update.release()
        # Swap front and back buffers
        self._save_snapshot()
        glfw.swap_buffers(self.window.handle)
        glfw.poll_events()

    # called by external threads
    def acquire_snapshot(self):
        
#         self.ss_update.acquire()
#         self.ss_ready.acquire()
#         self.ss_update.release()

        self.param_lock.acquire()
        img = self.snapshot.copy()
        self.param_lock.release()
        return img
        
    def _save_snapshot(self):
#         glfw.swap_buffers(self.window.handle)
        prev_buff_read = glGetIntegerv(GL_READ_BUFFER)
        glReadBuffer(GL_BACK)
        glFinish()
        buff = glReadPixels(self.window.width, 0, self.window.width, 
                         self.window.height, GL_RGB, GL_UNSIGNED_BYTE)
        glReadBuffer(prev_buff_read)
#         glfw.swap_buffers(self.window.handle)
        im = Image.fromstring(mode="RGB", data=buff, 
                              size=(self.window.width, self.window.height))
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        im = im.convert("L")
        self.snapshot = im
    
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

    def run(self):
#         draw_projected_shadows()
        
        self.init()
        self._init_finished_lock.release()
#         self.optimize()
        while not glfw.window_should_close(self.window.handle):
            self.param_lock.acquire()
            glDrawBuffer(GL_BACK)
            self.draw(None)
            self.param_lock.release()
            err = glGetError()
            if err != GL_NO_ERROR: print "Encountered a glError:", err
        glfw.terminate()
        pass

def _main():
    renderer = Renderer()
    renderer.start()
    from shadow_optim import MyGUI
    gui = MyGUI(renderer)
    gui.run()
    return

if __name__ == "__main__":
    print "----------start of main---------"
    _main()
    print "---------end of main-------------"
