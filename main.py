import pygame as pg
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import random
import math

RES = (640, 480)

objects = {}
cameraPos = [0, 0, 0]
cameraOrientation = [0, 0, 0]
move_speed = 0.3
mouse_sensitivity = 0.2
FOV = 90
FOVY = (FOV/RES[0])*RES[1]

pg.init()
pg.event.set_grab(False)
pg.mouse.set_visible(True)


def get_forward_vector(pitch, yaw):
    # Convert pitch and yaw from degrees to radians
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # Calculate forward direction
    x = math.cos(pitch_rad) * math.sin(yaw_rad)
    y = math.sin(pitch_rad)
    z = math.cos(pitch_rad) * math.cos(yaw_rad)
    
    return [x, y, z]  # Z is often inverted in 3D engines

def get_right_vector(yaw):
    yaw_rad = math.radians(yaw)
    x = math.sin(yaw_rad - math.pi / 2)
    z = math.cos(yaw_rad - math.pi / 2)
    return [x, 0, z]


def get_up_vector(pitch, yaw):
    forward = get_forward_vector(pitch, yaw)
    right = get_right_vector(yaw)
    up = cross(right, forward)
    # Optionally normalize the up vector:
    length = math.sqrt(sum(i*i for i in up))
    if length == 0:
        return [0, 1, 0]  # fallback up vector
    return [i / length for i in up]

def rotate_point_around_point(point, center, pitch_deg, yaw_deg, roll_deg):
    """
    Rotate a 3D point around an arbitrary center point by given Euler angles (degrees).
    
    Parameters:
        point (tuple/list): (x, y, z) coordinates of the point to rotate.
        center (tuple/list): (cx, cy, cz) coordinates of the center point to rotate around.
        pitch_deg (float): Rotation angle around X-axis in degrees.
        yaw_deg (float): Rotation angle around Y-axis in degrees.
        roll_deg (float): Rotation angle around Z-axis in degrees.
    
    Returns:
        tuple: Rotated (x, y, z) coordinates.
    """
    # Translate point to origin based on center
    x, y, z = point
    cx, cy, cz = center
    x -= cx
    y -= cy
    z -= cz
    
    # Convert degrees to radians
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)
    
    # Precompute sines and cosines
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    
    # Rotation matrices
    R_x = [
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp]
    ]
    
    R_y = [
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ]
    
    R_z = [
        [cr, -sr, 0],
        [sr, cr, 0],
        [0, 0, 1]
    ]
    
    # Multiply R_y and R_z: R_yz = R_y * R_z
    R_yz = [[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            R_yz[i][j] = sum(R_y[i][k]*R_z[k][j] for k in range(3))
    
    # Multiply R_x and R_yz: R = R_x * (R_y * R_z)
    R = [[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            R[i][j] = sum(R_x[i][k]*R_yz[k][j] for k in range(3))
    
    # Apply rotation matrix to translated point
    x_rot = R[0][0]*x + R[0][1]*y + R[0][2]*z
    y_rot = R[1][0]*x + R[1][1]*y + R[1][2]*z
    z_rot = R[2][0]*x + R[2][1]*y + R[2][2]*z
    
    # Translate back
    x_rot += cx
    y_rot += cy
    z_rot += cz
    
    return (x_rot, y_rot, z_rot)

def renderObjects(self):
    for triangles in objects.values():
        for triangle in triangles["Triangles"]:
            new_points = []
            for point in triangle:
                x, y, z, a = *rotate_point_around_point(point[:3], cameraPos, *cameraOrientation), point[3]
                x -= cameraPos[0]
                y -= cameraPos[1]
                z -= cameraPos[2]

                # x diff
                ax = x / ((z * math.tan(math.radians((FOV/2))) - z * math.tan(math.radians((-FOV/2))))/2)
                new_points.append(ax)

                # y diff
                ay = y / ((z * math.tan(math.radians((FOVY/2))) - z * math.tan(math.radians((-FOVY/2))))/2)
                new_points.append(ay)



                new_points.append(a)

            createTriangle(self, *new_points)

                



class App:
    def __init__(self):
        # initialize python
        pg.init()
        self.screen = pg.display.set_mode(RES, pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()

        # initialize OpenGL
        glClearColor(0.1, 0.2, 0.2, 1)
        self.shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")
        glUseProgram(self.shader)
        self.right_mouse_held = False

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

        createCube("a", (10,0,20), (5, 5, 5), (0, 0, 0))
        createCube("b", (-10,0,20), (5, 5, 5), (45, 45, 45))
        # createTween("a", (20,0,20), (5, 5, 5), (0, 0, 0), 5)
        print(Tweens)
        
        while running:
            global cameraPos
            # check events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 3:
                    self.right_mouse_held = True
                    pg.event.set_grab(True)
                    pg.mouse.set_visible(False)

                elif event.type == pg.MOUSEBUTTONUP and event.button == 3:
                    self.right_mouse_held = False
                    pg.event.set_grab(False)
                    pg.mouse.set_visible(True)

            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT)
            
            if random.randint(1, 10) == 1:
                print(cameraPos)
                print(cameraOrientation)

            performTweens()

            renderObjects(self)
            # createTriangle(self, *(1, 1), *(-1, -1), *(-1, -0))

            keys = pg.key.get_pressed()

            # Movement
            forward = get_forward_vector(*cameraOrientation[:2])
            # up = get_up_vector(*cameraOrientation[:2])
            right = get_right_vector(cameraOrientation[1])
            
            if keys[pg.K_w]:
                cameraPos = [cameraPos[i] + forward[i] * move_speed for i in range(3)]
            if keys[pg.K_s]:
                cameraPos = [cameraPos[i] - forward[i] * move_speed for i in range(3)]
            if keys[pg.K_a]:
                cameraPos = [cameraPos[i] + right[i] * move_speed for i in range(3)]
            if keys[pg.K_d]:
                cameraPos = [cameraPos[i] - right[i] * move_speed for i in range(3)]
            # if keys[pg.K_q]:
            #     cameraPos = [cameraPos[i] + up[i] * move_speed for i in range(3)]
            # if keys[pg.K_e]:
            #     cameraPos = [cameraPos[i] - up[i] * move_speed for i in range(3)]

            
            # Mouse look (while right-click is held)
            if self.right_mouse_held:
                dx, dy = pg.mouse.get_rel()
                cameraOrientation[1] -= dx * mouse_sensitivity  # Yaw
                cameraOrientation[0] -= dy * mouse_sensitivity  # Pitch

                # Clamp pitch to avoid flipping
                cameraOrientation[0] = max(-89, min(89, cameraOrientation[0]))
                

            font = pg.font.SysFont(None, 24)
            debug_text = font.render(f"Pos: {cameraPos}, Orient: {cameraOrientation}", True, (255, 255, 255))
            self.screen.blit(debug_text, (20, 20))
            
            pg.display.flip()

            # timing
            self.clock.tick(60)
        
        self.quit()

    def quit(self):

        # self.triangle.destroy()
        # glDeleteProgram(self.shader)

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

def createTriangle(self, ax, ay, a, bx, by, b, cx, cy, c):
    # vert = list(self.triangle.vertices)

    # for i in range(0, len(vert), 6):
    #     # Update position (first 3 values): range [-1, 1]
    #     for j in range(3):
    #         delta = random.uniform(-0.01, 0.01)
    #         vert[i + j] = self.clamp(vert[i + j] + delta, -1.0, 1.0)
    #     # Update color (next 3 values): range [0, 1]
    #     for j in range(3, 6):
    #         delta = random.uniform(-0.1, 0.1)
    #         vert[i + j] = self.clamp(vert[i + j] + delta, 0.0, 1.0)

    vert = (
        ax, ay, 0.0, a, a, a,
        bx, by, 0.0, b, b, b,
        cx, cy, 0.0, c, c, c,
    )

    new_triangle = Triangle(vert)
    self.triangle = new_triangle

    glUseProgram(self.shader)
    glBindVertexArray(self.triangle.vao)
    glDrawArrays(GL_TRIANGLES, 0, self.triangle.vertex_count)


def normalize(v):
    length = math.sqrt(sum(coord ** 2 for coord in v))
    if length == 0:
        return (0, 0, 0)
    return tuple(coord / length for coord in v)

def vector_sub(a, b):
    return tuple(ai - bi for ai, bi in zip(a, b))

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    )

def rotate_vertex(v, orientation):
    x, y, z = v
    rx, ry, rz = orientation

    # Rotate around X-axis
    cos_x, sin_x = math.cos(rx), math.sin(rx)
    y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x

    # Rotate around Y-axis
    cos_y, sin_y = math.cos(ry), math.sin(ry)
    x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y

    # Rotate around Z-axis
    cos_z, sin_z = math.cos(rz), math.sin(rz)
    x, y = x * cos_z - y * sin_z, x * sin_z + y * cos_z

    return (x, y, z)

def createCube(name, origin, dimensions, orientation=(0, 0, 0)):
    orientation = tuple([math.radians(i) for i in orientation])
    global cameraPos

    cx, cy, cz = origin
    camx, camy, camz = cameraPos
    w, h, d = dimensions
    hw, hh, hd = w / 2, h / 2, d / 2

    # Define raw vertex positions relative to the center (origin)
    raw_vertices = [
        (-hw, -hh, -hd),  # 0
        ( hw, -hh, -hd),  # 1
        ( hw,  hh, -hd),  # 2
        (-hw,  hh, -hd),  # 3
        (-hw, -hh,  hd),  # 4
        ( hw, -hh,  hd),  # 5
        ( hw,  hh,  hd),  # 6
        (-hw,  hh,  hd),  # 7
    ]

    # Rotate and translate vertices
    rotated_vertices = []
    for x, y, z in raw_vertices:
        rx, ry, rz = rotate_vertex((x, y, z), orientation)
        world_x, world_y, world_z = cx + rx, cy + ry, cz + rz
        rotated_vertices.append((world_x, world_y, world_z, 0.5))

    # # Calculate top and bottom Y for shading
    # y_values = [v[1] for v in rotated_vertices]
    # min_y = min(y_values)
    # max_y = max(y_values)

    # # Apply shading
    # shaded_vertices = []
    # for x, y, z in rotated_vertices:
    #     if max_y == min_y:
    #         shade = 0  # avoid division by zero
    #     else:
    #         shade = (y - min_y) / (max_y - min_y)
    #     shaded_vertices.append((x, y, z, shade))

    # Define faces using vertex indices
    face_indices = [
        [0, 1, 2, 3],  # front
        [4, 5, 6, 7],  # back
        [0, 1, 5, 4],  # bottom
        [3, 2, 6, 7],  # top
        [1, 2, 6, 5],  # right
        [0, 3, 7, 4],  # left
    ]

    # Compute face center distance to camera for sorting
    face_info = []
    for face in face_indices:
        pts = [rotated_vertices[i][:3] for i in face]
        center = (
            sum(p[0] for p in pts) / 4,
            sum(p[1] for p in pts) / 4,
            sum(p[2] for p in pts) / 4,
        )
        dx, dy, dz = center[0] - camx, center[1] - camy, center[2] - camz
        dist_sq = dx * dx + dy * dy + dz * dz
        face_info.append((dist_sq, face))

    # Sort faces farthest to nearest (for painter's algorithm)
    face_info.sort(reverse=True, key=lambda x: x[0])

    # Convert faces to triangles
    triangles = []
    for _, face in face_info:
        i0, i1, i2, i3 = face
        triangles.append([rotated_vertices[i0], rotated_vertices[i1], rotated_vertices[i2]])
        triangles.append([rotated_vertices[i0], rotated_vertices[i2], rotated_vertices[i3]])

    objects[name] = {}
    objects[name]["Info"] = ["cube", origin, dimensions, orientation]
    objects[name]["Triangles"] = triangles



Tweens = []
def createTween(obj, new_pos, new_size, new_rotation, time):
    frames = 60*time
    pos = tuple((a - b)/frames for a, b in zip(new_pos, objects[obj]["Info"][1]))
    size = tuple((a - b)/frames for a, b in zip(new_size, objects[obj]["Info"][2]))
    rot = tuple((a - b)/frames for a, b in zip(new_rotation, objects[obj]["Info"][3]))
    Tweens.append([obj, pos, size, rot, frames])

def performTweens():
    if len(Tweens) != 0:
        a = 0
        for i, v in enumerate(Tweens):
            name, pos, size, orientate, times = v
            createCube(name, 
                    tuple(a + b for a, b in zip(pos, objects[name]["Info"][1])), 
                    tuple(a + b for a, b in zip(size, objects[name]["Info"][2])),
                    tuple(a + b for a, b in zip(orientate, objects[name]["Info"][3])))
            Tweens[i-a][-1] -= 1
            if Tweens[i-a][-1] == 0:
                del Tweens[i-a]
                a += 1
        



# print(objects)


if __name__ == "__main__":
    myApp = App()