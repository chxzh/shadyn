import glfw
import atb
import tools
from app import Application, Object
from OpenGL.GL import *
from ctypes import c_uint8, c_float, c_ushort, c_void_p
from math import pi
from cgkit.cgtypes import *
from OpenGL.raw.GL.ARB.vertex_buffer_object import GL_ARRAY_BUFFER_ARB
from OpenGL.GL.VERSION.GL_1_5 import glGenBuffers
from OpenGL.raw.GL.VERSION.GL_1_5 import glBindBuffer
import numpy as np
from mpl_toolkits.axisartist import floating_axes
from win32con import N_TMASK
from PIL import Image

def _main():
    draw_projected_shadows()
#     draw_a_few_cubes()
    return

def draw_projected_shadows():    
    flatten = lambda l: [u for t in l for u in t]
    c_array = lambda c_type: lambda l: (c_type*len(l))(*l)    
    look_at = lambda eye, at, up: mat4.lookAt(eye, 2*eye - at, up).inverse()
    def shadow_proj_mat(plane_normal, plane_point, light_pos):
        if type(plane_normal) == vec3:
            plane_normal = plane_normal.normalize()
        elif type(plane_normal) == np.array:
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
        else:
            raise TypeError("What the hell did you put in here as a normal??")
        n_t = np.array(plane_normal).reshape((1,3))
        L = np.array(light_pos).reshape((3,1))
        D = - plane_normal * plane_point
        ntL = np.dot(n_t, L)
        shad_mat = np.identity(4, float)
        shad_mat[0:3, 0:3] = L.dot(n_t) - (D + ntL ) * np.identity(3)
        shad_mat[0:3, 3:4] = (D+ntL) * L - L * ntL
        shad_mat[3:4, 0:3] = n_t
        shad_mat[3:4, 3:4] = - ntL
        return mat4(shad_mat.astype(np.float32).T.tolist())
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
    cube_obj = Object("../obj/cube.obj")
    
    # initialize VAO
    vao_handle = glGenVertexArrays(1)
    glBindVertexArray(vao_handle)
    
    # bind buffers
    # indices buffer
    i_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, i_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (c_ushort * len(cube_obj.indices))(*cube_obj.indices),
                 GL_STATIC_DRAW)
    # vertices buffer
    v_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    v_flatten = flatten(cube_obj.vertices)
    glBufferData(GL_ARRAY_BUFFER,
                 (c_float * len(v_flatten))(*v_flatten),
                 GL_STATIC_DRAW)
    # normals buffer
    n_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, n_buffer)
    n_flatten = flatten(cube_obj.normals)
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
    model_mat = mat4(1.0)
    model_mat.scale(vec3(0.5))
    model_mat.rotate(pi / 3, vec3(1.0, 0.5, 1.7))
    model_mat.translate((0.5, 0, 0))
    # TODO: fix this stupid left-handed coord lookAt func
    view_mat = look_at(vec3(-1, 2, 5),
                           vec3(0, 0, 0),
                           vec3(0, 1, 0))
    view_mat_top = look_at(vec3(0, 4, 0),
                           vec3(0, 0, 0),
                           vec3(0, 0, -1))
    proj_mat = mat4.perspective(45, 4./3, 0.1, 100)
    model_view_inv = (view_mat * model_mat).inverse()
    light_pos = vec3(3,3,0)
    MVP = proj_mat * view_mat * model_mat
    V_loc = glGetUniformLocation(program_handle, "V")
    glUniformMatrix4fv(V_loc, 1, GL_FALSE, view_mat.toList())
    light_pos_loc = glGetUniformLocation(program_handle, "LightPosition_worldspace")
    glUniform3f(light_pos_loc, light_pos.x, light_pos.y, light_pos.z)
    MVP_loc = glGetUniformLocation(program_handle, "MVP")
    M_loc = glGetUniformLocation(program_handle, "M")
    MVint_loc = glGetUniformLocation(program_handle, "MVint")
    
    floor_model_mat = mat4.translation((0,-0.51,0))*mat4.scaling((5,0.1,5))
    floor_MVP = proj_mat*view_mat*floor_model_mat
    floor_MVinv = (view_mat*floor_model_mat).inverse()
    
    # initialize shadow projection program
    shadow_program_handle = tools.load_program("../shader/shadowProjectionShading.v.glsl",
                                               "../shader/shadowProjectionShading.f.glsl")
    glUseProgram(shadow_program_handle)
    shadow_MsVP_loc = glGetUniformLocation(shadow_program_handle, "MsVP")
    VP_mat = proj_mat * view_mat;
    VP_mat_top = proj_mat * view_mat_top;
    shaject_mat = shadow_proj_mat(vec3(0,1,0), vec3(0,-0.45,0), light_pos)
    glUniform3f(glGetUniformLocation(shadow_program_handle, "shadowColor"),
                 0.0, 0.0, 0.0) # black shadow
    shadow_v_loc = glGetAttribLocation(shadow_program_handle, "coord3d")
    glEnableVertexAttribArray(shadow_v_loc)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    glVertexAttribPointer(shadow_v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
    
    basic_program_handle = tools.load_program("../shader/basic.v.glsl",
                                               "../shader/basic.f.glsl")
    glUseProgram(basic_program_handle)
    basic_mvp_loc = glGetUniformLocation(basic_program_handle, "mvp")
    floor_basic_mvp = proj_mat*view_mat_top*floor_model_mat
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
    def draw(fake_param=0):        
        # Render here
        # Make the window's context current
        shaject_mat = shadow_proj_mat(vec3(0,1,0), vec3(0,-0.45,0), light_pos)
        
        glfw.make_context_current(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                
        # draw the scene          
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE)
        glViewport(0,0,width,height)
        glUseProgram(program_handle)   
        glUniform3f(light_pos_loc, light_pos.x, light_pos.y, light_pos.z)     
        glUniformMatrix4fv(MVint_loc, 1, GL_TRUE, model_view_inv.toList())
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, model_mat.toList())
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, MVP.toList())
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None);        
        glUniformMatrix4fv(MVint_loc, 1, GL_TRUE, floor_MVinv.toList())
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, floor_model_mat.toList())
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, floor_MVP.toList())                        
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None)        
        glDisable(GL_CULL_FACE)
        glUseProgram(shadow_program_handle)
        glUniformMatrix4fv(shadow_MsVP_loc, 1, GL_FALSE, (VP_mat*shaject_mat*model_mat).toList())              
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None)
        
        
        glViewport(width, 0, width, height)
        
        glDisable(GL_CULL_FACE)               
        glUseProgram(basic_program_handle)                       
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None)  
        glUseProgram(shadow_program_handle)
        glUniformMatrix4fv(shadow_MsVP_loc, 1, GL_FALSE, (VP_mat_top*shaject_mat*model_mat).toList())              
        
#         glUniformMatrix4fv(shadow_M_loc, 1, GL_FALSE, model_mat.toList())
#         glUniformMatrix4fv(shadow_VP_loc, 1, GL_FALSE, VP_mat_top.toList())                 
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
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
    def optim_obj(x):
        # x[0] the light z
        light_pos.z = x[0]
        draw()
        image = get_image()
        xor = imath.eval("a^b", a=image, b=image_obj)
        res = sum(xor.getdata())
        print res
        return res
    draw()
    r = raw_input("dfsadfsa")
    from scipy import optimize
    res = optimize.minimize(fun=optim_obj, x0=0.0, method='Powell', callback=draw)
    print "__end of optimization__"
    print res
#     light_pos.z = res.x[0]
#     print light_pos.z
#     im = get_image()
#     im.save("target.png")
#     im.show()
    while not glfw.window_should_close(window):
        draw()
    glfw.terminate();
    pass

def draw_a_few_cubes():    
    flatten = lambda l: [u for t in l for u in t]
    c_array = lambda c_type: lambda l: (c_type*len(l))(*l)
    look_at = lambda eye, at, up: mat4.lookAt(eye, 2*eye - at, up).inverse()
    if not glfw.init():
        return -1;
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 480, "scene", None, None);
#     window2 = glfw.create_window(640, 480, "floor", None, None);
    if window == None:
        glfw.terminate()
        return -1
    # Loop until the user closes the window
    glfw.make_context_current(window)
    glClearColor(0.0, 0.0, 0.2, 1.0)
#     glfw.make_context_current(window2)
#     glClearColor(0.0, 0.2, 0.2, 1.0)    
#     glfw.make_context_current(window)
    program_handle = tools.load_program("../shader/flatShading.v.glsl",
                                        "../shader/flatShading.f.glsl")
    glUseProgram(program_handle)
    cube_obj = Object("../obj/cube.obj")
    
    # initialize VAO
    vao_handle = glGenVertexArrays(1)
    glBindVertexArray(vao_handle)
    
    # bind buffers
    # indices buffer
    i_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, i_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (c_ushort * len(cube_obj.indices))(*cube_obj.indices),
                 GL_STATIC_DRAW)
    # vertices buffer
    v_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    v_flatten = flatten(cube_obj.vertices)
    glBufferData(GL_ARRAY_BUFFER,
                 (c_float * len(v_flatten))(*v_flatten),
                 GL_STATIC_DRAW)
    # normals buffer
    n_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, n_buffer)
    n_flatten = flatten(cube_obj.normals)
    glBufferData(GL_ARRAY_BUFFER,
                 (c_float * len(n_flatten))(*n_flatten),
                 GL_STATIC_DRAW)
    
    # attributes initializing
    vert_loc = glGetAttribLocation(program_handle, "vertexPosition_modelspace")
    glEnableVertexAttribArray(vert_loc)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    glVertexAttribPointer(vert_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
    # TODO: fix the existing attribute unable to retrieve problem
    norm_loc = 0# glGetAttribLocation(program_handle, "vertexNormal_modelspace")
    glEnableVertexAttribArray(norm_loc)
    glBindBuffer(GL_ARRAY_BUFFER, n_buffer)
    glVertexAttribPointer(norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
    
    # uniforms    
    model_mat = mat4(1.0)
    model_mat.scale(vec3(0.5))
    model_mat.rotate(pi / 2, vec3(1.0, 0.5, 1.7))
    model_mat.translate((0.5, 0, 0))
    view_mat = look_at(vec3(0, 2, -5),
                           vec3(0, 0, 0),
                           vec3(0, 1, 0))
    proj_mat = mat4.perspective(45, 4./3, 0.1, 100)
    model_view_inv = (view_mat * model_mat).inverse()
    light_pos = vec3(3,3,3)
    MVP = proj_mat * view_mat * model_mat
    V_loc = glGetUniformLocation(program_handle, "V")
    glUniformMatrix4fv(V_loc, 1, GL_FALSE, view_mat.toList())
    light_pos_loc = glGetUniformLocation(program_handle, "LightPosition_worldspace")
    glUniform3f(light_pos_loc, light_pos.x, light_pos.y, light_pos.z)
    MVP_loc = glGetUniformLocation(program_handle, "MVP")
    M_loc = glGetUniformLocation(program_handle, "M")
    MVint_loc = glGetUniformLocation(program_handle, "MVint")
    
    floor_model_mat = mat4.translation((0,-0.5,0))*mat4.scaling((5,0.1,5))
    floor_MVP = proj_mat*view_mat*floor_model_mat
    floor_MVinv = (view_mat*floor_model_mat).inverse()
    
    flatten_biasmat = [ 0.5, 0.0, 0.0, 0.0,
                        0.0, 0.5, 0.0, 0.0,
                        0.0, 0.0, 0.5, 0.0,
                        0.5, 0.5, 0.5, 1.0]
    flatten_biasmat = (c_float*16)(*flatten_biasmat)
    bias_loc = glGetUniformLocation(program_handle, "DepthBiasMVP")
    glUniformMatrix4fv(bias_loc, 1, GL_FALSE, flatten_biasmat)
    shadow_map_loc = glGetUniformLocation(program_handle, "shadowMap")
    
    # initializing other stuff
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glEnable(GL_CULL_FACE)
    
    # initializing shadow mapping
    depth_program_handle = tools.load_program("../shader/DepthRTT.v.glsl",
                                              "../shader/DepthRTT.f.glsl")
    depth_MVP_loc = glGetUniformLocation(depth_program_handle, "depthMVP")
    depth_framebuffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, depth_framebuffer)
    depth_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_tex)
    glTexImage2D(GL_TEXTURE_2D, # target
                 0,             # level
                 GL_DEPTH_COMPONENT16,  # internal format
                 1024,          # width
                 1024,          # height
                 0,             # border
                 GL_DEPTH_COMPONENT,     # format
                 GL_FLOAT,      # type
                 None)          # data pointer
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);    
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_tex, 0)
    glDrawBuffer(GL_NONE)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("framebuffer is not okay")
    depth_proj_mat = mat4.perspective(45, 1, 1, 50)
    depth_view_mat = look_at(light_pos, vec3(0), vec3(0,1,0))
    glUseProgram(depth_program_handle)
    depth_vert_loc = glGetAttribLocation(depth_program_handle, 
                                         "vertexPosition_modelspace")
    glEnableVertexAttribArray(depth_vert_loc)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    print "vertices buffer handle:", v_buffer
    glVertexAttribPointer(depth_vert_loc,   # attribute handle                          
                          3,                # size
                          GL_FLOAT,         # type
                          GL_FALSE,         # unnormalized
                          0,                # stride
                          None              # array buffer offset
                          )
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, i_buffer)
    
    # initialize the texture monitor
    quad_vertices = [   -1.0, -1.0, 0.0,   1.0, -1.0, 0.0,   -1.0,  1.0, 0.0,\
                        -1.0,  1.0, 0.0,   1.0, -1.0, 0.0,    1.0,  1.0, 0.0]
    quad_v_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, quad_v_buffer)
    glBufferData(GL_ARRAY_BUFFER, (c_float*(len(quad_vertices)))(*quad_vertices), GL_STATIC_DRAW)
    quad_program_handle = tools.load_program("../shader/Passthrough.v.glsl",
                                             "../shader/SimpleTexture.f.glsl")
    tex_loc = glGetUniformLocation(quad_program_handle, "texture");

    while not glfw.window_should_close(window):
        # Render here
        # Make the window's context current
        glfw.make_context_current(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # draw the shadow map
        glBindFramebuffer(GL_FRAMEBUFFER, depth_framebuffer)
        glViewport(0,0,1024,1024)
        # TODO: see if the coming lines can be moved outside
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(depth_program_handle)
        depth_MVP = depth_proj_mat * depth_view_mat * model_mat
        glUniformMatrix4fv(depth_MVP_loc, 1, GL_FALSE, depth_MVP.toList())
        
        glEnableVertexAttribArray(depth_vert_loc)
        glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
        glVertexAttribPointer(depth_vert_loc,   # attribute handle
                              3,                # size
                              GL_FLOAT,         # type
                              GL_FALSE,         # unnormalized
                              0,                # stride
                              None              # array buffer offset
                              )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, i_buffer)
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None)
        
        
        # draw the scene          
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0,0,640,480)
        glUseProgram(program_handle)        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth_tex)
        glUniform1i(shadow_map_loc, 0)
        glUniformMatrix4fv(MVint_loc, 1, GL_TRUE, model_view_inv.toList())
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, model_mat.toList())
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, MVP.toList())
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None);        
        glUniformMatrix4fv(MVint_loc, 1, GL_TRUE, floor_MVinv.toList())
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, floor_model_mat.toList())
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, floor_MVP.toList())                        
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None)
        
        # texture monitor
        glViewport(0, 0, 256, 256)
        glUseProgram(quad_program_handle)
        # Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth_tex)
        # Set our "renderedTexture" sampler to user Texture Unit 0
        glUniform1i(tex_loc, 0)
        # 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, quad_v_buffer)
        glVertexAttribPointer(
            0,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                  # size
            GL_FLOAT,           # type
            GL_FALSE,           # normalized?
            0,                  # stride
            None            # array buffer offset
            )
        # Draw the triangle !
        # You have to disable GL_COMPARE_R_TO_TEXTURE above in order to see anything !
        glDrawArrays(GL_TRIANGLES, 0, 6) # 2*3 indices starting at 0 -> 2 triangles
        glDisableVertexAttribArray(0)
        # Swap front and back buffers 
        glfw.swap_buffers(window)
        
#         glfw.make_context_current(window2)
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         
#         glfw.swap_buffers(window2)        
        
        # Poll for and process events
        glfw.poll_events()
    glfw.terminate();
    pass

def draw_a_white_cube():
    flatten = lambda l: [u for t in l for u in t]
    c_array = lambda c_type: lambda l: (c_type*len(l))(*l)
    if not glfw.init():
        return -1;
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 480, "white cube", None, None);
    if window == None:
        glfw.terminate()
        return -1
    # Make the window's context current
    glfw.make_context_current(window)
    # Loop until the user closes the window
    glClearColor(0.0, 0.0, 0.2, 1.0)
    program_handle = tools.load_program("../shader/cube.v.glsl", "../shader/cube.f.glsl")
    glUseProgram(program_handle)
    cube_obj = Object("../obj/cube.obj")
    
    # initialize VAO
    vao_handle = glGenVertexArrays(1)
    glBindVertexArray(vao_handle)
    
    # bind buffers
    # indices buffer
    i_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, i_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (c_ushort * len(cube_obj.indices))(*cube_obj.indices),
                 GL_STATIC_DRAW)
    # vertices buffer
    v_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    v_flatten = flatten(cube_obj.vertices)
    glBufferData(GL_ARRAY_BUFFER,
                 (c_float * len(v_flatten))(*v_flatten),
                 GL_STATIC_DRAW)
    # normals buffer
    n_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, n_buffer)
    n_flatten = flatten(cube_obj.normals)
    glBufferData(GL_ARRAY_BUFFER,
                 (c_float * len(n_flatten))(*n_flatten),
                 GL_STATIC_DRAW)
    
    # attributes initializing
    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    # uniforms    
    c_mat4 = lambda mat4: (c_float * 16)(*(mat4.toList()))
    model_mat = mat4(1.0)
    model_mat.scale(vec3(0.5))
    model_mat.rotate(pi / 3, vec3(1.0, 1.0, 0))
    model_mat.translate((0.5, 0, 0))
    view_mat = mat4.lookAt(vec3(0, 0, -5),
                           vec3(0, 0, 0))
    proj_mat = mat4.perspective(45, 4./3, 0.1, 100)
    mvp = proj_mat * view_mat * model_mat
    c_mvp = c_mat4(mvp)
    mvp_id = glGetUniformLocation(program_handle, "mvp")
    glUniformMatrix4fv(mvp_id, 1, GL_FALSE, c_mvp)
    
    
    # initializing other stuff
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    
    while not glfw.window_should_close(window):
        # Render here
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Swap front and back buffers 
        
        # bind buffer to vao
#         glEnableVertexAttribArray(n_buffer)
#         glBindBuffer(GL_ARRAY_BUFFER, n_buffer)
#         glVertexAttribPointer(v_buffer, 3, GL_FLOAT, GL_FALSE, 0, 0)
          
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, i_buffer)
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None);

        glfw.swap_buffers(window)
        # Poll for and process events
        glfw.poll_events()
    glfw.terminate();

def draw_a_triangle():
    
    if not glfw.init():
        return -1;
    # Create a windowed mode window and its OpenGL context
    
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(1024, 768, "Triangle", None, None);
    if window == None:
        glfw.terminate()
        return -1
    # Make the window's context current
    
    glfw.make_context_current(window)
#     glfw.Experimental = True
    glClearColor(0.0, 0.1, 0.2, 1.0)
    
    flatten = lambda l: [u for t in l for u in t]
    vertices = [(-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (0.0, 1.0, 0.0)]
    indices = range(3)
    vao_handle = glGenVertexArrays(1)
    glBindVertexArray(vao_handle)
    program_handle = tools.load_program("../shader/simple.v.glsl",
                                        "../shader/simple.f.glsl")
    
    f_vertices = flatten(vertices)
    c_vertices = (c_float*len(f_vertices))(*f_vertices)
    v_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    glBufferData(GL_ARRAY_BUFFER, c_vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, v_buffer)
    glVertexAttribPointer(0,
        #glGetAttribLocation(program_handle, "vertexPosition_modelspace"),
        3, GL_FLOAT, False, 0, None)
    
    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Render here
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(program_handle)
        
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glDisableVertexAttribArray(vao_handle)
        
        # Swap front and back buffers 
        glfw.swap_buffers(window)
        # Poll for and process events
        glfw.poll_events()
    glfw.terminate();
    
    pass

def just_a_window():
    if not glfw.init():
        return -1;
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 480, "Hello World", None, None);
    if window == None:
        glfw.terminate()
        return -1
    # Make the window's context current
    glfw.make_context_current(window)
    # Loop until the user closes the window
    glClearColor(0.0, 0.0, 0.2, 1.0)
    while not glfw.window_should_close(window):
        # Render here
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Swap front and back buffers 
        glfw.swap_buffers(window)
        # Poll for and process events
        glfw.poll_events()
    glfw.terminate();

if __name__ == "__main__":
    print "---starting main()---"
    _main()
    print "---end of main()---"
