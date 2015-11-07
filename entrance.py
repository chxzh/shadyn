import glfw
def main():

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
    while not glfw.window_should_close(window):
        # Render here
        # Swap front and back buffers 
        glfw.swap_buffers(window)
        # Poll for and process events
        glfw.poll_events()
    glfw.terminate();

if __name__ == "__main__":
    main()