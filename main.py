import pygame as pg
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import random

class App:
    def __init__(self):
        # initialize python
        pg.init()
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()

        # initialize OpenGL
        glClearColor(0.1, 0.2, 0.2, 1)
        self.shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")
        glUseProgram(self.shader)
        self.triangle = Triangle()
        self.mainLoop()

    def createShader(self, vertexFilePath, fragmentFilePath):
        with open(vertexFilePath, "r") as f:
            vertex_src = f.readlines()

        with open(fragmentFilePath, "r") as f:
            fragment_src = f.readlines()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

        return shader
    
    def clamp(self, value, min_val, max_val):
        return max(min_val, min(max_val, value))

    def mainLoop(self):
        running = True
        
        while running:
            # check events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT)

            vert = list(self.triangle.vertices)

            for i in range(0, len(vert), 6):
                # Update position (first 3 values): range [-1, 1]
                for j in range(3):
                    delta = random.uniform(-0.01, 0.01)
                    vert[i + j] = self.clamp(vert[i + j] + delta, -1.0, 1.0)
                # Update color (next 3 values): range [0, 1]
                for j in range(3, 6):
                    delta = random.uniform(-0.1, 0.1)
                    vert[i + j] = self.clamp(vert[i + j] + delta, 0.0, 1.0)


            new_triangle = Triangle(tuple(vert))
            self.triangle = new_triangle

            glUseProgram(self.shader)
            glBindVertexArray(self.triangle.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.triangle.vertex_count)

            pg.display.flip()

            # timing
            self.clock.tick(60)
        
        self.quit()

    def quit(self):

        self.triangle.destroy()
        glDeleteProgram(self.shader)

        pg.quit()

class Triangle:
    def __init__(self, vert_choice = None):
        # x, y, z, r, g, b
        if vert_choice != None:
            self.vertices = vert_choice
        else:
            self.vertices = (
                -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
                0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
                0.0,  0.5, 0.0, 0.0, 0.0, 1.0,
            )

        self.vertices = np.array(self.vertices, dtype = np.float32)

        self.vertex_count = 3

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


if __name__ == "__main__":
    myApp = App()