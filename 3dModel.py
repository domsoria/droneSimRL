import pygame
from pygame.locals import *
from math import cos, sin, pi

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

quadcopter_pos = [0, 0, 0]
# Questo elenco memorizza le posizioni passate del drone
quadcopter_path = []

# Definizione dei componenti che rappresentano il quadricottero
components = {
    "body": {
        "size": [0.3, 0.1, 0.3],
        "color": [1, 0, 0]
    },
    "arm1": {
        "size": [0.8, 0.05, 0.05],
        "color": [0, 0, 1],
        "position": [0.2, 0, 0.2],
        "rotation": 45
    },
    "arm2": {
        "size": [0.8, 0.05, 0.05],
        "color": [0, 0, 1],
        "position": [-0.2, 0, -0.2],
        "rotation": 45
    },
    "arm3": {
        "size": [0.8, 0.05, 0.05],
        "color": [0, 0, 1],
        "position": [0.2, 0, -0.2],
        "rotation": -45
    },
    "arm4": {
        "size": [0.8, 0.05, 0.05],
        "color": [0, 0, 1],
        "position": [-0.2, 0, 0.2],
        "rotation": -45
    },
    "propeller1": {
        "size": [0.2, 0.01, 0.2],
        "color": [0, 1, 0],
        "position": [0.6, 0, 0.6]
    },
    "propeller2": {
        "size": [0.2, 0.01, 0.2],
        "color": [0, 1, 0],
        "position": [-0.6, 0, -0.6]
    },
    "propeller3": {
        "size": [0.2, 0.01, 0.2],
        "color": [0, 1, 0],
        "position": [0.6, 0, -0.6]
    },
    "propeller4": {
        "size": [0.2, 0.01, 0.2],
        "color": [0, 1, 0],
        "position": [-0.6, 0, 0.6]
    }
}

def Quadricottero():
    glPushMatrix()  # Save current matrix
    glTranslatef(quadcopter_pos[0], quadcopter_pos[1], quadcopter_pos[2])  # Move quadcopter
    for component, attributes in components.items():
        glPushMatrix()  # Push Matrix on stack
        glRotatef(attributes.get("rotation", 0), 0, 1, 0)  # Rotate the Matrix
        drawCube(attributes["size"], attributes["color"], attributes.get("position", [0, 0, 0]))
        glPopMatrix()  # Pop Matrix from stack
    glPopMatrix()  # Restore matrix to previous state (before quadcopter translation)



def drawCube(size, color, position):
    vertices = [
        [size[0]/2, -size[1]/2, -size[2]/2], 
        [size[0]/2, size[1]/2, -size[2]/2],
        [-size[0]/2, size[1]/2, -size[2]/2], 
        [-size[0]/2, -size[1]/2, -size[2]/2], 
        [size[0]/2, -size[1]/2, size[2]/2], 
        [size[0]/2, size[1]/2, size[2]/2], 
        [-size[0]/2, -size[1]/2, size[2]/2], 
        [-size[0]/2, size[1]/2, size[2]/2]
    ]
    vertices = [[v[0]+position[0], v[1]+position[1], v[2]+position[2]] for v in vertices]

    edges = [
        (0,1), (0,3), (0,4), (2,1), (2,3), (2,7),
        (6,3), (6,4), (6,7), (5,1), (5,4), (5,7)
    ]

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glColor3fv(color)
            glVertex3fv(vertices[vertex])
    glEnd()

def move_quadcopter(forward, backward, up, down, left, right):
    global quadcopter_pos
    if forward:
        quadcopter_pos[2] -= 0.1
    if backward:
        quadcopter_pos[2] += 0.1
    if up:
        quadcopter_pos[1] += 0.1
    if down:
        quadcopter_pos[1] -= 0.1
    if left:
        quadcopter_pos[0] -= 0.1
    if right:
        quadcopter_pos[0] += 0.1

def draw_target(position, radius, color):
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor3fv(color)
    
    # Draw a "circle" in 3D
    glBegin(GL_POLYGON)
    for i in range(100):
        glVertex3f(radius * cos(2*pi*i/100), radius * sin(2*pi*i/100), 0.0)
    glEnd()

    glPopMatrix()


# Funzione per verificare se il drone si trova nella stessa posizione di un altro oggetto
def is_colliding(pos1, pos2, tolerance=0.2):
    return abs(pos1[0] - pos2[0]) < tolerance and abs(pos1[1] - pos2[1]) < tolerance and abs(pos1[2] - pos2[2]) < tolerance


def main():
    global quadcopter_pos, quadcopter_path

    pygame.init()
    display = (1024,768)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    # Initial camera position
    glTranslatef(0.0, -1.0, -5)

    # Questa è la posizione dell'obiettivo o dell'oggetto con cui il drone potrebbe collidere
    target_position = [1, 1, 1]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()
        move_quadcopter(keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_SPACE], keys[pygame.K_LSHIFT], keys[pygame.K_a], keys[pygame.K_d])

        # Memorizza il percorso del drone
        quadcopter_path.append(quadcopter_pos[:])  # Fai attenzione a copiare la lista!

        # Verifica se il drone sta collidendo con l'obiettivo
        if is_colliding(quadcopter_pos, target_position):
            print("Il drone ha colliso con l'obiettivo!")

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Quadricottero()
        draw_target(target_position, 0.2, (1,0,0))  # Draw a red sphere at position [2, 2, 2]
        pygame.display.flip()
        pygame.time.wait(10)

main()
