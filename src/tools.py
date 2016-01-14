from OpenGL.GL import *
import glfw
def load_obj(filepath):
    pass

def load_shader(shader_code, shader_type):
    shader_handle = glCreateShader(shader_type)
    glShaderSource(shader_handle, shader_code)
    glCompileShader(shader_handle)
    if glGetShaderiv(shader_handle, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader_handle))
    return shader_handle

def load_program(vert_shader_path, frag_shader_path):
    program_handle = glCreateProgram()
    shader_paths = [(vert_shader_path, GL_VERTEX_SHADER),
                   (frag_shader_path, GL_FRAGMENT_SHADER)]
    shader_list = []
    for src_path, shader_type in shader_paths:
        with open(src_path) as src_file:
            src = ""
            for line in src_file.readlines():
                src += line
            shader_handle = load_shader(src, shader_type)
            shader_list.append(shader_handle)
            glAttachShader(program_handle, shader_handle)
    glLinkProgram(program_handle)
    for shader_handle in shader_list:
        glDetachShader(program_handle, shader_handle)
        glDeleteShader(shader_handle)
    return program_handle

def _main():
    pass

if __name__ == "__main__":    
    print "start of _main()"
    _main()
    print "end of _main()"
    pass
