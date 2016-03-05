from OpenGL.GL import *
import glfw
import os
from datetime import datetime as dt
from astropy.units import bar
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


def get_fname(root=None):
    if not root:
        root = os.getcwd()
    elif not os.path.exists(root):
        os.mkdir(root)
    time_stamp_str = dt.now().strftime("%m-%d-%H-%M-%S-%y")
    path = root + '\\' + time_stamp_str
    return path, time_stamp_str

def ring_bisector_generator_closure(upper=360):
    # upper is not included, i.e. [0, upper)
    def hue_generator():
        yield 0
        i = 1
        bar = 2
        while True:
            hue = upper * ((i - bar / 2) * 2 + 1) / bar
            i += 1
            if i >= bar: bar *= 2
            yield hue
    return hue_generator

hue_generator = ring_bisector_generator_closure(360)

from random import randint
def random_bright_color_generator():
    upper = 6*64
    hue_gen = ring_bisector_generator_closure(upper)()
#     offset = 0
    offset = randint(0, upper - 1)
    while True:
        hue = hue_gen.next() + offset
        hue = hue % upper
        color = [0., 0., 0.]
        i = hue * 3 / upper
        color[i] = 1.
        diff = hue - upper / 6 * (2*i+1)
        if diff < 0: j = -1
        else: j = 1
        j = (j + i) % 3
        color[j] = float(abs(diff)) * 6/ upper
        yield color  

def _main():
    pass

if __name__ == "__main__":    
    print "start of _main()"
    _main()
    print "end of _main()"
    pass
