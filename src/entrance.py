import glfw
import atb
from app import Application
from OpenGL.GL import *
def main():
    app = Application()
    app.run()
    print "end of main"
    return

def basic_glfw_example():
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
    while not glfw.window_should_close(window):
        # Render here
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Swap front and back buffers 
        glfw.swap_buffers(window)
        # Poll for and process events
        glfw.poll_events()
    glfw.terminate();

if __name__ == "__main__":
    main()