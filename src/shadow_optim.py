import glfw
import numpy as np
from OpenGL.GL import *
import tools
from app import *
from ctypes import c_uint8, c_float, c_ushort, c_void_p
from math import pi, cos, sin
from cgkit.cgtypes import *
from PIL import Image
import cma
from scipy import optimize

def draw_projected_shadows():
    flatten = lambda l: [u for t in l for u in t]
    c_array = lambda c_type: lambda l: (c_type * len(l))(*l)
    look_at = lambda eye, at, up: mat4.lookAt(eye, 2 * eye - at, up).inverse()
    def shadow_proj_mat(plane_normal, plane_point, light_pos):
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
    class Item:
        pass
    cube = Item()  # currently just a temporary holder of attributes and uniforms
    if not glfw.init():
        return -1;
    # Create a windowed mode window and its OpenGL context
    width, height = (640, 480)
    window = glfw.create_window(width * 2, height, "scene", None, None);
#     window2 = glfw.create_window(640, 480, "floor", None, None);
    if window == None:
        glfw.terminate()
        return -1
    # Loop until the user closes the window
    glfw.make_context_current(window)
    glClearColor(0.0, 0.0, 0.2, 1.0)
    program_handle = tools.load_program("../shader/standardShading.v.glsl",
                                        "../shader/standardShading.f.glsl")
    glUseProgram(program_handle)
    cube.obj = Object("../obj/cube.obj")

    # initialize VAO
    vao_handle = glGenVertexArrays(1)
    glBindVertexArray(vao_handle)

    # bind buffers
    # indices buffer
    i_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, i_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (c_ushort * len(cube.obj.indices))(*cube.obj.indices),
                 GL_STATIC_DRAW)
    # vertices buffer
    v_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    v_flatten = flatten(cube.obj.vertices)
    glBufferData(GL_ARRAY_BUFFER,
                 (c_float * len(v_flatten))(*v_flatten),
                 GL_STATIC_DRAW)
    # normals buffer
    n_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, n_buffer)
    n_flatten = flatten(cube.obj.normals)
    glBufferData(GL_ARRAY_BUFFER,
                 (c_float * len(n_flatten))(*n_flatten),
                 GL_STATIC_DRAW)

    # attributes initializing
    vert_loc = glGetAttribLocation(program_handle, "vertexPosition_modelspace")
    glEnableVertexAttribArray(vert_loc)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    glVertexAttribPointer(vert_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
    # TODO: fix the existing attribute unable to retrieve problem
    norm_loc = glGetAttribLocation(program_handle, "vertexNormal_modelspace")
    glEnableVertexAttribArray(norm_loc)
    glBindBuffer(GL_ARRAY_BUFFER, n_buffer)
    glVertexAttribPointer(norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

    # uniforms
    cube.model_mat = mat4(1.0)
    cube.model_mat.scale(vec3(0.5))
    cube.model_mat.rotate(pi / 3, vec3(1.0, 0.5, 1.7))
    cube.position = vec3(0.5, 0, 1)
    cube.model_mat.translate(cube.position)
    # TODO: fix this stupid left-handed coord lookAt func
    view_mat = look_at(vec3(-1, 2, 5),
                           vec3(0, 0, 0),
                           vec3(0, 1, 0))
    view_mat_top = look_at(vec3(0, 4, 0),
                           vec3(0, 0, 0),
                           vec3(0, 0, -1))
    proj_mat = mat4.perspective(45, 4. / 3, 0.1, 100)
    model_view_inv = (view_mat * cube.model_mat).inverse()
#     light_pos = vec3(2,1,0)
#     light_pos = vec3(2,2,2)
    light_pos = vec3(3, 3, 3)
    V_loc = glGetUniformLocation(program_handle, "V")
    glUniformMatrix4fv(V_loc, 1, GL_FALSE, view_mat.toList())
    light_pos_loc = glGetUniformLocation(program_handle, "LightPosition_worldspace")
    glUniform3f(light_pos_loc, light_pos.x, light_pos.y, light_pos.z)
    MVP_loc = glGetUniformLocation(program_handle, "MVP")
    M_loc = glGetUniformLocation(program_handle, "M")
    MVint_loc = glGetUniformLocation(program_handle, "MVint")

    floor_model_mat = mat4.translation((0, -0.51, 0)) * mat4.scaling((5, 0.1, 5))
    floor_MVP = proj_mat * view_mat * floor_model_mat
    floor_MVinv = (view_mat * floor_model_mat).inverse()

    # initialize shadow projection program
    shadow_program_handle = tools.load_program("../shader/shadowProjectionShading.v.glsl",
                                               "../shader/shadowProjectionShading.f.glsl")
    glUseProgram(shadow_program_handle)
    shadow_MsVP_loc = glGetUniformLocation(shadow_program_handle, "MsVP")
    VP_mat = proj_mat * view_mat;
    VP_mat_top = proj_mat * view_mat_top;
    shaject_mat = shadow_proj_mat(vec3(0, 1, 0), vec3(0, -0.45, 0), light_pos)
    glUniform3f(glGetUniformLocation(shadow_program_handle, "shadowColor"),
                 0.0, 0.0, 0.0)  # black shadow
    shadow_v_loc = glGetAttribLocation(shadow_program_handle, "coord3d")
    glEnableVertexAttribArray(shadow_v_loc)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    glVertexAttribPointer(shadow_v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

    basic_program_handle = tools.load_program("../shader/basic.v.glsl",
                                               "../shader/basic.f.glsl")
    glUseProgram(basic_program_handle)
    basic_mvp_loc = glGetUniformLocation(basic_program_handle, "mvp")
    floor_basic_mvp = proj_mat * view_mat_top * floor_model_mat
    glUniformMatrix4fv(basic_mvp_loc, 1, GL_FALSE, floor_basic_mvp.toList())
    basic_v_loc = glGetAttribLocation(basic_program_handle, "coord3d")
    glEnableVertexAttribArray(basic_v_loc)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    glVertexAttribPointer(basic_v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

    # initializing other stuff
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glEnable(GL_CULL_FACE)
    image_obj = Image.open("../img/target.png")
    image_obj = image_obj.convert("L")
    def set_param(x):
        cube.model_mat = mat4(1.0)
        cube.model_mat.scale(vec3(0.5))
        cube.model_mat.rotate(pi / 3, vec3(1.0, 0.5, 1.7))
#         cube.model_mat.rotate(x[5], vec3(cos(x[4])*cos(x[3]), sin(x[4]), cos(x[4])*sin(x[3])))
        cube.model_mat.translate((x[0], x[1], x[2]))
        return

    def draw(x):
        set_param(x)
        # Render here
        # Make the window's context current
        shaject_mat = shadow_proj_mat(vec3(0, 1, 0), vec3(0, -0.45, 0), light_pos)

        glfw.make_context_current(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE)
        glViewport(0, 0, width, height)
        glUseProgram(program_handle)
        glUniform3f(light_pos_loc, light_pos.x, light_pos.y, light_pos.z)
        model_view_inv = (view_mat * cube.model_mat).inverse()
        glUniformMatrix4fv(MVint_loc, 1, GL_TRUE, model_view_inv.toList())
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, cube.model_mat.toList())
        MVP = proj_mat * view_mat * cube.model_mat
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, MVP.toList())
        glDrawElements(GL_TRIANGLES, len(cube.obj.indices),
                        GL_UNSIGNED_SHORT, None);
        glUniformMatrix4fv(MVint_loc, 1, GL_TRUE, floor_MVinv.toList())
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, floor_model_mat.toList())
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, floor_MVP.toList())
        glDrawElements(GL_TRIANGLES, len(cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        glDisable(GL_CULL_FACE)
        glUseProgram(shadow_program_handle)
        glUniformMatrix4fv(shadow_MsVP_loc, 1, GL_FALSE, (VP_mat * shaject_mat * cube.model_mat).toList())
        glDrawElements(GL_TRIANGLES, len(cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)


        glViewport(width, 0, width, height)

        glDisable(GL_CULL_FACE)
        glUseProgram(basic_program_handle)
        glDrawElements(GL_TRIANGLES, len(cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        glUseProgram(shadow_program_handle)
        glUniformMatrix4fv(shadow_MsVP_loc, 1, GL_FALSE, (VP_mat_top * shaject_mat * cube.model_mat).toList())

#         glUniformMatrix4fv(shadow_M_loc, 1, GL_FALSE, model_mat.toList())
#         glUniformMatrix4fv(shadow_VP_loc, 1, GL_FALSE, VP_mat_top.toList())
        glDrawElements(GL_TRIANGLES, len(cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        # Swap front and back buffers
        glfw.swap_buffers(window)
        glfw.poll_events()

    def get_image():
        glfw.swap_buffers(window)
        b = glReadPixels(width, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        glfw.swap_buffers(window)
        im = Image.fromstring(mode="RGB", size=(width, height), data=b)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        im = im.convert("L")
        return im

    from PIL import ImageMath as imath
    def optim_obj_xor(x):
        draw(x)
        image = get_image()
        xor = imath.eval("a^b", a=image, b=image_obj)
        res = sum(xor.getdata())
        print x, res
        return res

    X, Y = np.arange(width).reshape(1, width), np.arange(height).reshape(height, 1)
    def get_sec_moment(image):
        # image should be a gray scale Image object
        img = 1 - np.array(image.getdata()) / 128  # turn white to 0 and black to 1
        # using 128 in case of gray
        img = img.astype(np.int8)
        img = img.reshape(height, width)
        M_00 = float(img.sum())
        M_10 = (X * img).sum()
        M_01 = (img * Y).sum()
        m_10 = M_10 / M_00 if M_00 else 0
        m_01 = M_01 / M_00 if M_00 else 0
        X_offset = X - m_10
        Y_offset = Y - m_01
        M_20 = ((X_offset ** 2) * img).sum() / M_00 if M_00 else 0
        M_02 = (img * (Y_offset ** 2)).sum() / M_00 if M_00 else 0
        M_11 = (X_offset * img * Y_offset).sum() / M_00 if M_00 else 0
        return np.array([M_20, M_11, M_02])

    Mt_2 = get_sec_moment(image_obj)
    def optim_obj_sec_moment(x):
        draw(x)
        image = get_image()
        M_2 = get_sec_moment(image)
        res = ((Mt_2 - M_2) ** 2).sum()
        print res, x
        return res


    class Optim:
        def __init__(self):
            self.error_func = lambda:None
            self.weight = None
            self.method = ""
            self.params = None

        def set_weight(self, *coefficients):
            self.weight = np.array(coefficients)


    optim = Optim()


    def get_jac(func, delta, x0):
        # let func be the air-function and delta as the uniform delta for gradient
        len_x = len(x0)
        def jac(x):
            fx = func(x)
            grad = np.zeros(len_x)
            for i in range(len_x):
                x_t = np.zeros(len_x)
                x_t[i] = delta
                fx_t = func(x + x_t)
                grad[i] = fx_t - fx
            return grad / delta
        return jac

    x0 = np.array([0.5, 0.5, 0])
#     print Mt_2
#     print optim_obj_sec_moment(x0)
#     raw_input("return to continue")
    x_res = x0
    draw(x0)
    import cma
    res = cma.fmin(objective_function=optim_obj_sec_moment,
             x0=x0,
             sigma0=1)
    x_res = res[0]
    optimize = False
    if optimize:
        from scipy import optimize
        res = optimize.minimize(fun=optim_obj_sec_moment, x0=x0, method='BFGS',
                                callback=None, jac=get_jac(optim_obj_sec_moment, 0.005, x0),
                                bounds=((0, 2.5), (0, 2.5), (None, None)))
        print "__end of optimization__"
        print res
        x_res = res.x

#     light_pos.z = res.x[0]
#     print light_pos.z
#     im = get_image()
#     im.save("target.png")
#     im.show()
    while not glfw.window_should_close(window):
        draw(x_res)
    glfw.terminate();
    pass

from PySide.QtGui import *
from PySide.QtCore import *
from threading import Thread, Lock
import sys

qt_app = QApplication(sys.argv)
class MyGUI(QWidget):
    def __init__(self, renderer):
        QWidget.__init__(self)
        self.renderer = renderer
        self._init_layout()
        self._init_components()
        self._optimizer = None

    def _init_layout(self):
        self.setMinimumSize(400, 185)
        self.setMaximumWidth(600)

    def _init_components(self):
        self.launch_button = QPushButton("launch", self)
        self.launch_button.resize(100, 20)
        self.launch_button.move(5, 5)
        QObject.connect(self.launch_button, SIGNAL("clicked()"), self._on_launch)

    def _on_launch(self):
        self._optimizer = Optimizer(self.renderer)
        self._optimizer.start()


    def run(self):
        self.show()
        qt_app.exec_()


class Optimizer(Thread):
    def __init__(self, renderer):
        Thread.__init__(self)
        self.green_light = Lock()
        self.renderer = renderer
    
    def run(self):
        # collect params
        # build optimizer
        # wrap optimizer with green_light
        wrapped = self._wrap_eval(self.renderer.optimize)
        wrapped(self.renderer.get_param())
        pass
        
    def _wrap_eval(self, func):
        '''
        wrapping the evaluation function to be controllable by locks
        '''
        def wrapped(x):
            self.green_light.acquire()
            self.green_light.release()
            return func(x)
        return wrapped


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
        pass

    class _Window:  # a windows attribute holder
        pass

    def init(self):
        # windows initialization
        self.window = self._Window()
        self.window.width, self.window.height = (640, 480)
        self.window.handle = glfw.create_window(self.window.width * 2, self.window.height, "scene", None, None)
        if self.window.handle == None:
            glfw.terminate()
            raise RuntimeError("GLFW cannot create a window.")
        glfw.make_context_current(self.window.handle)
        glClearColor(0.0, 0.0, 0.2, 1.0)

        # default blinn-phong shader loading
        self.program_handle = tools.load_program("../shader/standardShading.v.glsl",
                                        "../shader/standardShading.f.glsl")
        glUseProgram(self.program_handle)

        self.cube = self._Item()
        self.cube.obj = Object("../obj/cube.obj")

        # initialize VAO
        self.vao_handle = glGenVertexArrays(1)
        glBindVertexArray(self.vao_handle)

        # bind buffers
        # indices buffer
        self.cube.i_buffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cube.i_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     (c_ushort * len(self.cube.obj.indices))(*self.cube.obj.indices),
                     GL_STATIC_DRAW)
        # vertices buffer
        self.cube.v_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.v_buffer)
        v_flatten = self.flatten(self.cube.obj.vertices)
        glBufferData(GL_ARRAY_BUFFER,
                     (c_float * len(v_flatten))(*v_flatten),
                     GL_STATIC_DRAW)
        # normals buffer
        self.cube.n_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.n_buffer)
        n_flatten = self.flatten(self.cube.obj.normals)
        glBufferData(GL_ARRAY_BUFFER,
                     (c_float * len(n_flatten))(*n_flatten),
                     GL_STATIC_DRAW)

        # attributes initializing
        self.cube.vert_loc = glGetAttribLocation(self.program_handle,
                                                 "vertexPosition_modelspace")
        glEnableVertexAttribArray(self.cube.vert_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.v_buffer)
        glVertexAttribPointer(self.cube.vert_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
        # TODO: fix the existing attribute unable to retrieve problem
        self.cube.norm_loc = glGetAttribLocation(self.program_handle,
                                                 "vertexNormal_modelspace")
        glEnableVertexAttribArray(self.cube.norm_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.n_buffer)
        glVertexAttribPointer(self.cube.norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        # cube uniforms
        self.cube.model_mat = mat4(1.0)
        self.cube.model_mat.scale(vec3(0.5))
        self.cube.model_mat.rotate(pi / 3, vec3(1.0, 0.5, 1.7))
        self.cube.position = vec3(0.5, 0, 1)
        self.cube.model_mat.translate(self.cube.position)

        # camera initializing
        self.cam_obs = self._Item()  # the camera for human observation
        self.cam_obs.view_mat = self.look_at(vec3(-1, 2, 5),
                                   vec3(0, 0, 0),
                                   vec3(0, 1, 0))
        self.cam_cap = self._Item()  # the camera to capture shadow
        self.cam_cap.view_mat = self.look_at(vec3(0, 4, 0),
                                   vec3(0, 0, 0),
                                   vec3(0, 0, -1))
        self.cam_cap.proj_mat = self.cam_obs.proj_mat = mat4.perspective(45, 4. / 3, 0.1, 100)
        self.model_view_inv = (self.cam_obs.view_mat * self.cube.model_mat).inverse()
    #     light_pos = vec3(2,1,0)
    #     light_pos = vec3(2,2,2)
        self.light_pos = vec3(3, 3, 3)
        self.cam_obs.V_loc = glGetUniformLocation(self.program_handle, "V")
        glUniformMatrix4fv(self.cam_obs.V_loc, 1, GL_FALSE, self.cam_obs.view_mat.toList())
        self.light_pos_loc = glGetUniformLocation(self.program_handle,
                                                  "LightPosition_worldspace")
        glUniform3f(self.light_pos_loc,
                    self.light_pos.x, self.light_pos.y, self.light_pos.z)
        self.cube.MVP_loc = glGetUniformLocation(self.program_handle, "MVP")
        self.cube.MVP = mat4(1)
        self.cube.M_loc = glGetUniformLocation(self.program_handle, "M")
        self.MVint_loc = glGetUniformLocation(self.program_handle, "MVint")

        # init the floor
        self.floor = self._Item()
        self.floor.model_mat = mat4.translation((0, -0.51, 0)) * mat4.scaling((5, 0.1, 5))
        self.floor.MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * self.floor.model_mat
        self.floor.MVinv = (self.cam_obs.view_mat * self.floor.model_mat).inverse()

        # initialize shadow projection program
        self.shadow_program_handle = tools.load_program("../shader/shadowProjectionShading.v.glsl",
                                                   "../shader/shadowProjectionShading.f.glsl")
        glUseProgram(self.shadow_program_handle)
        self.shadow = self._Item()
        self.shadow.MsVP_loc = glGetUniformLocation(self.shadow_program_handle, "MsVP")
        self.shadow.VP_mat = self.cam_obs.proj_mat * self.cam_obs.view_mat;
        self.shadow.VP_mat_top = self.cam_cap.proj_mat * self.cam_cap.view_mat;
        self.shadow.shaject_mat = self.shadow_proj_mat(vec3(0, 1, 0), vec3(0, -0.45, 0), self.light_pos)
        glUniform3f(glGetUniformLocation(self.shadow_program_handle, "shadowColor"),
                     0.0, 0.0, 0.0)  # black shadow
        self.shadow.v_loc = glGetAttribLocation(self.shadow_program_handle, "coord3d")
        glEnableVertexAttribArray(self.shadow.v_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.v_buffer)
        glVertexAttribPointer(self.shadow.v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        # init the shader to draw the basic shadow
        self.basic_program_handle = tools.load_program("../shader/basic.v.glsl",
                                                   "../shader/basic.f.glsl")
        glUseProgram(self.basic_program_handle)
        self.basic_mvp_loc = glGetUniformLocation(self.basic_program_handle, "mvp")
        self.floor_basic_mvp = self.cam_cap.proj_mat * self.cam_cap.view_mat * self.floor.model_mat
        glUniformMatrix4fv(self.basic_mvp_loc, 1, GL_FALSE, self.floor_basic_mvp.toList())
        self.basic_v_loc = glGetAttribLocation(self.basic_program_handle, "coord3d")
        glEnableVertexAttribArray(self.basic_v_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.v_buffer)
        glVertexAttribPointer(self.basic_v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        # initializing other stuff
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        self.image_target = Image.open("../img/target.png")
        self.image_target = self.image_target.convert("L")
        
        self._X = np.arange(self.window.width).reshape(1,self.window.width)
        self._Y = np.arange(self.window.height).reshape(self.window.height,1)
        
        self.Mt_2 = self._get_sec_moment(self.image_target)   

    def __init__(self):
        Thread.__init__(self)
        if not glfw.init():
            raise RuntimeError("Cannot start up GLFW")
        self.flatten = lambda l: [u for t in l for u in t]
        self.c_array = lambda c_type: lambda l: (c_type * len(l))(*l)
        # TODO: rewrite this stupid expedient look_at function
        self.look_at = lambda eye, at, up: mat4.lookAt(eye, 2 * eye - at, up).inverse()
#         self.init()
        self.ss_update = Lock()
        self.ss_ready = Lock()
        self.ss_ready.acquire()
        self.snapshot = None
        self.param_lock = Lock()

    def draw(self, x):
#         self.set_param(x)
        # Render here
        # Make the window's context current
        self.shadow.shaject_mat = self.shadow_proj_mat(vec3(0, 1, 0), 
                                                       vec3(0, -0.45, 0), 
                                                       self.light_pos)

        glfw.make_context_current(self.window.handle)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE)
        glViewport(0, 0, self.window.width, self.window.height)
        glUseProgram(self.program_handle)
        glUniform3f(self.light_pos_loc, *self.light_pos)
        model_view_inv = (self.cam_obs.view_mat * self.cube.model_mat).inverse()
        glUniformMatrix4fv(self.MVint_loc, 1, GL_TRUE, self.model_view_inv.toList())
        glUniformMatrix4fv(self.cube.M_loc, 1, GL_FALSE, self.cube.model_mat.toList())
        self.cube.MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * self.cube.model_mat
        glUniformMatrix4fv(self.cube.MVP_loc, 1, GL_FALSE, self.cube.MVP.toList())
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None);
        glUniformMatrix4fv(self.MVint_loc, 1, GL_TRUE, self.floor.MVinv.toList())
        glUniformMatrix4fv(self.cube.M_loc, 1, GL_FALSE, self.floor.model_mat.toList())
        glUniformMatrix4fv(self.cube.MVP_loc, 1, GL_FALSE, self.floor.MVP.toList())
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        glDisable(GL_CULL_FACE)
        glUseProgram(self.shadow_program_handle)
        glUniformMatrix4fv(self.shadow.MsVP_loc, 1, GL_FALSE,
               (self.shadow.VP_mat * self.shadow.shaject_mat * self.cube.model_mat).toList())
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)


        glViewport(self.window.width, 0, self.window.width, self.window.height)

        glDisable(GL_CULL_FACE)
        glUseProgram(self.basic_program_handle)
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        glUseProgram(self.shadow_program_handle)
        glUniformMatrix4fv(self.shadow.MsVP_loc, 1, GL_FALSE,
               (self.shadow.VP_mat_top * self.shadow.shaject_mat * self.cube.model_mat).toList())

#         glUniformMatrix4fv(shadow_M_loc, 1, GL_FALSE, model_mat.toList())
#         glUniformMatrix4fv(shadow_VP_loc, 1, GL_FALSE, VP_mat_top.toList())
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        # Swap front and back buffers
        glfw.swap_buffers(self.window.handle)
        if self.ss_update.locked():
            self._save_snapshot()
            self.ss_ready.release()
        glfw.poll_events()

    def optimize(self, x=None):
        if x == None:
            x = self.cube.position
#         res = optimize.minimize(fun=self._optim_obj_sec_moment, x0=x, method='BFGS', 
#                                 callback=None, bounds=((0, 2.5), (0, 2.5), (None, None)),
#                                 jac=self._get_jac(self._optim_obj_sec_moment, 0.005, x))
#         x_res = res.x
        res = cma.fmin(objective_function=self._optim_obj_sec_moment, 
             x0=x,
             sigma0=1)    
        x_res = res[0]
        self.set_param(x_res)
        print "optm ends"
    
    def _get_jac(self, func, delta, x0):
        # let func be the air-function and delta as the uniform delta for gradient
        len_x = len(x0)
        def jac(x):
            fx = func(x)
            grad = np.zeros(len_x)        
            for i in range(len_x):
                x_t = np.zeros(len_x)
                x_t[i] = delta
                fx_t = func(x+x_t)
                grad[i] = fx_t - fx
            return grad / delta
        return jac
    
    def _optim_obj_sec_moment(self, x):
        self.set_param(x)
        self.ss_update.acquire()
        self.ss_ready.acquire()
        self.ss_update.release()
        image = self.snapshot
        M_2 = self._get_sec_moment(image)
        res = ((self.Mt_2 - M_2)**2).sum()
        print res, x
        return res
    
    def _save_snapshot(self):                   
        glfw.swap_buffers(self.window.handle)
        buff = glReadPixels(self.window.width, 0, self.window.width, 
                         self.window.height, GL_RGB, GL_UNSIGNED_BYTE)
        glfw.swap_buffers(self.window.handle)
        im = Image.fromstring(mode="RGB", data=buff, 
                              size=(self.window.width, self.window.height))
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        im = im.convert("L")
        self.snapshot = im
    
    def _get_sec_moment(self, image):
        # image should be a gray scale Image object
        img = 1 - np.array(image.getdata()) / 128 # turn white to 0 and black to 1
        # using 128 in case of gray
        img = img.astype(np.int8)
        img = img.reshape(self.window.height, self.window.width)
        M_00 = float(img.sum())        
        M_10 = (self._X * img).sum()
        M_01 = (img * self._Y).sum()
        m_10 = M_10 / M_00 if M_00 else 0
        m_01 = M_01 / M_00 if M_00 else 0
        X_offset = self._X-m_10
        Y_offset = self._Y-m_01
        M_20 = ((X_offset**2)*img).sum() / M_00 if M_00 else 0
        M_02 = (img*(Y_offset**2)).sum() / M_00 if M_00 else 0
        M_11 = (X_offset*img*Y_offset).sum() / M_00 if M_00 else 0
        return np.array([M_20, M_11, M_02])

    def set_param(self, x):
        self.param_lock.acquire()
        self.cube.model_mat = mat4(1.0)
        self.cube.model_mat.scale(vec3(0.5))
        self.cube.model_mat.rotate(pi / 3, vec3(1.0, 0.5, 1.7))
#         self.cube.model_mat.rotate(x[5], vec3(cos(x[4])*cos(x[3]), sin(x[4]), cos(x[4])*sin(x[3])))
        self.cube.model_mat.translate((x[0], x[1], x[2]))
        self.cube.position = vec3(*x)
        self.param_lock.release()
        return
    
    def get_param(self):
        return np.array(self.cube.position)

    def set_optimizor(self):
        pass

    def to_close():
        return False

    def run(self):
#         draw_projected_shadows()
        self.init()
#         self.optimize()
        while not glfw.window_should_close(self.window.handle):
            self.param_lock.acquire()
            self.draw(None)
            self.param_lock.release()
        glfw.terminate()
        pass

def _main():
    renderer = Renderer()
    renderer.start()
    gui = MyGUI(renderer)
    gui.run()
    return

if __name__ == "__main__":
    print "----------start of main---------"
    _main()
    print "---------end of main-------------"
