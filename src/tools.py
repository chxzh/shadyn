from OpenGL.GL import *
import glfw
def load_obj(filepath):
    pass

def load_shader(shader_code, shader_type):
    shader_handle = glCreateShader(shader_type)
    glShaderSource(shader_handle, shader_code)
    glCompileShader(shader_handle)
    return shader_handle

def load_program(vert_shader_path, frag_shader_path):
    program_handle = glCreateProgram()
    shader_list = [(vert_shader_path, GL_VERTEX_SHADER),
                   (frag_shader_path, GL_FRAGMENT_SHADER)]
    for src_path, shader_type in shader_list:
        with open(src_path) as src_file:
            src = ""
            for line in src_file.readlines():
                src += line
            shader_handle = load_shader(src, shader_type)
            glAttachShader(program_handle, shader_handle)
    glLinkProgram(program_handle)
    return program_handle

def _main():
    if not glfw.init():
        return -1;
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 480, "Hello World", None, None);
    if window == None:
        glfw.terminate()
        return -1
    #Make the window's context current
    glfw.make_context_current(window)
    # Loop until the user closes the window
    glClearColor(0.0, 0.0, 0.2, 1.0)

    with open('../shader/cube.v.glsl') as shader_file:
        shader_code = ""
        for line in shader_file.readlines():
            shader_code += line
        shader_handle = load_shader(shader_code, GL_VERTEX_SHADER)
        print shader_handle
    while not glfw.window_should_close(window):
        # Render here
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Swap front and back buffers 
        glfw.swap_buffers(window)
        # Poll for and process events
        glfw.poll_events()
    glfw.terminate();
if __name__ == "__main__":
    
    print "start of _main()"
    _main()
    print "end of _main()"
    pass
