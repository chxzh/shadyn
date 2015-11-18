# --------------------------------------------------------------------------------
# Copyright (c) 2013 Mack Stone. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# --------------------------------------------------------------------------------

"""
Modern OpenGL with python.
render a color triangle with pyopengl using freeglut.

@author: Mack Stone
"""

import ctypes

import numpy
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *

VERTEX_SHADER = """
#version 330

layout (location=0) in vec4 position;
layout (location=1) in vec4 color;

smooth out vec4 theColor;

void main()
{
    gl_Position = position;
    theColor = color;
}
"""

FRAGMENT_SHADER = """
#version 330

smooth in vec4 theColor;
out vec4 outputColor;

void main()
{
    outputColor = theColor;
}
"""

shaderProgram = None
VAO = None

def initialize():
    global VERTEX_SHADER
    global FRAGMENT_SHADER
    global shaderProgram
    global VAO
    # compile shaders and program
    vertexShader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragmentShader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    shaderProgram = shaders.compileProgram(vertexShader, fragmentShader)

    # triangle position and color
    vertexData = numpy.array([0.0, 0.5, 0.0, 1.0,
                            0.5, -0.366, 0.0, 1.0,
                            -0.5, -0.366, 0.0, 1.0,
                            1.0, 0.0, 0.0, 1.0,
                            0.0, 1.0, 0.0, 1.0,
                            0.0, 0.0, 1.0, 1.0, ],
                            dtype=numpy.float32)

    # create VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # create VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL_STATIC_DRAW)

    # enable array and set up data
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
    # the last parameter is a pointer
    # python donot have pointer, have to using ctypes
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(48))

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

def render():
    global shaderProgram
    global VAO
    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # active shader program
    glUseProgram(shaderProgram)

    glBindVertexArray(VAO)

    # draw triangle
    glDrawArrays(GL_TRIANGLES, 0, 3)

    glBindVertexArray(0)
    glUseProgram(0)

    glutSwapBuffers()

def main():
    # init freeglut
    glutInit([])

    # make a window
    glutInitContextVersion(3, 3)
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
    glutInitContextProfile(GLUT_CORE_PROFILE)

    glutInitWindowSize(640, 480)
    glutCreateWindow("pyopengl with glut")

    initialize()

    glutDisplayFunc(render)

    glutMainLoop()

if __name__ == '__main__':
    main()
