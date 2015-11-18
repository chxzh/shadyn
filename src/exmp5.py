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
render a color triangle with pyopengl using PySide/PyQt4.

@author: Mack Stone
"""

import ctypes

import numpy
from OpenGL.GL import *
from OpenGL.GL import shaders
from PySide import QtGui, QtOpenGL
#from PyQt4 import QtGui, QtOpenGL

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

class MyWidget(QtOpenGL.QGLWidget):
    def initializeGL(self):
        glViewport(0, 0, self.width(), self.height())

        # compile shaders and program
        vertexShader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fragmentShader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.shaderProgram = shaders.compileProgram(vertexShader, fragmentShader)

        # triangle position and color
        vertexData = numpy.array([0.0, 0.5, 0.0, 1.0,
                                0.5, -0.366, 0.0, 1.0,
                                -0.5, -0.366, 0.0, 1.0,
                                1.0, 0.0, 0.0, 1.0,
                                0.0, 1.0, 0.0, 1.0,
                                0.0, 0.0, 1.0, 1.0, ],
                                dtype=numpy.float32)

        # create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # create VBO
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL_STATIC_DRAW)

        # enable array and set up data
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        # the last parameter is a pointer
        # python donot have pointer, have to use ctypes
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(48))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def paintGL(self):
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # active shader program
        glUseProgram(self.shaderProgram)

        glBindVertexArray(self.VAO)

        # draw triangle
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glBindVertexArray(0)
        glUseProgram(0)


def main():
    import sys

    app = QtGui.QApplication(sys.argv)

    glformat = QtOpenGL.QGLFormat()
    glformat.setVersion(3, 3)
    glformat.setProfile(QtOpenGL.QGLFormat.CoreProfile)
    w = MyWidget(glformat)
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
