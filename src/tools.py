from OpenGL.GL import *
def load_obj(filepath):
    pass

def load_shader(shader_code, shader_type):
    shader_handle = glCreateShader(shader_type)
    glShaderSource(shader_handle, shader_code)
    glCompileShader(shader_handle)
    return shader_handle

def load_program(vert_shader_path, frag_shader_path):
    pass

def _main():
    pass

if __name__ == "__main__()":
    _main()
    print "end of _main()"
    return
