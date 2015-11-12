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
    pass

if __name__ == "__main__":    
    print "start of _main()"
    _main()
    print "end of _main()"
    pass
