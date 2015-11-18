import glfw
import atb
import tools
from app import Application, Object
from OpenGL.GL import *
from ctypes import c_uint8, c_float, c_ushort, c_void_p
from math import pi
from cgkit.cgtypes import *
from OpenGL.raw.GL.ARB.vertex_buffer_object import GL_ARRAY_BUFFER_ARB

def _main():
    draw_a_few_cubes()
    return

def draw_a_few_cubes():    
    flatten = lambda l: [u for t in l for u in t]
    c_array = lambda c_type: lambda l: (c_type*len(l))(*l)
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
    program_handle = tools.load_program("../shader/StandardShading.v.glsl", "../shader/StandardShading.f.glsl")
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
    norm_loc = glGetAttribLocation(program_handle, "vertexNormal_modelspace")
    glEnableVertexAttribArray(norm_loc)
    glBindBuffer(GL_ARRAY_BUFFER, n_buffer)
    glVertexAttribPointer(norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
    
    # uniforms    
    c_mat4 = lambda mat4: (c_float * 16)(*(mat4.toList()))
    model_mat = mat4(1.0)
    model_mat.scale(vec3(0.5))
    model_mat.rotate(pi / 6, vec3(0.0, 1.0, 0.0))
    model_mat.translate((0.5, 0, 0))
    view_mat = mat4.lookAt(vec3(0, 0, -5),
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
    
    floor_model_mat = mat4.translation((0,-2,0))*mat4.scaling((5,1,5))
    floor_MVP = proj_mat*view_mat*floor_model_mat
    floor_MVinv = (view_mat*floor_model_mat).inverse()
    
    
    light_pos  = vec3(3,3,3)
    
    
    # initializing other stuff
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    
    while not glfw.window_should_close(window):
        # Render here
        # Make the window's context current
        glfw.make_context_current(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # bind buffer to vao
#         glEnableVertexAttribArray(n_buffer)
#         glBindBuffer(GL_ARRAY_BUFFER, n_buffer)
#         glVertexAttribPointer(v_buffer, 3, GL_FLOAT, GL_FALSE, 0, 0)
          
        
        glUniformMatrix4fv(MVint_loc, 1, GL_TRUE, model_view_inv.toList())
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, model_mat.toList())
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, MVP.toList())
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, i_buffer)
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None);
        
        glUniformMatrix4fv(MVint_loc, 1, GL_TRUE, floor_MVinv.toList())
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, floor_model_mat.toList())
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, floor_MVP.toList())                        
        glDrawElements(GL_TRIANGLES, len(cube_obj.indices),
                        GL_UNSIGNED_SHORT, None);
        
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
