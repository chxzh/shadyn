import glfw
import atb
from app import Application
from OpenGL.GL import *
def main():
    app = Application()
    app.run()
    print "end of main"
    return

if __name__ == "__main__":
    main()