import pygame
from pygame.locals import *
from math import cos, sin, pi
import pygame.display
from OpenGL.GLUT import *


from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    


action_mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3,'forward' : 4, 'backward': 5}  # Update this mapping based on your specific actions

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

# Funzione per muovere il drone basata sull'azione selezionata
def move_quadcopter(action_index):
    global quadcopter_pos
    action = action_space[action_index]
    
    if action == 'forward':
        quadcopter_pos[2] -= 0.1
    elif action == 'backward':
        quadcopter_pos[2] += 0.1
    elif action == 'up':
        quadcopter_pos[1] += 0.1
    elif action == 'down':
        quadcopter_pos[1] -= 0.1
    elif action == 'left':
        quadcopter_pos[0] -= 0.1
    elif action == 'right':
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



# Variabili globali per lo spazio delle azioni e lo spazio degli stati
action_space = ['up', 'down', 'left', 'right','forward', 'backward']
#state_space = list(np.arange(-5, 5.1, 0.1))  # Posizioni da -5 a 5 con incrementi di 0.1
state_space = list(np.round(np.arange(-5, 5.1, 0.1), 1))

# Dimensione dello stato e delle azioni
state_size = 3  # x, y, z coordinate
action_size = len(action_space)

# Creare e ottimizzare la rete neurale
qnetwork = QNetwork(state_size, action_size)
optimizer = optim.Adam(qnetwork.parameters(), lr=0.001)

#q_table = np.zeros((len(state_space), len(state_space), len(state_space), len(action_space)))

# Funzione di ricompensa
# Funzione di ricompensa
def reward(old_state, new_state, target):
    old_distance = np.sqrt(np.sum((np.array(old_state) - np.array(target))**2))
    new_distance = np.sqrt(np.sum((np.array(new_state) - np.array(target))**2))

    if is_colliding(new_state, target):
        return 100  # Ricompensa grande per aver raggiunto l'obiettivo
    elif new_distance < old_distance:
        return 1  # Ricompensa positiva per avvicinarsi all'obiettivo
    else:
        return -1  # Penalità per allontanarsi dall'obiettivo


# Funzione di scelta dell'azione
def choose_action(state, epsilon):
    state = torch.tensor(state, dtype=torch.float32)
    if np.random.uniform(0, 1) < epsilon:
        action_index = np.random.choice(len(action_space))
    else:
        with torch.no_grad():
            action_index = torch.argmax(qnetwork(state)).item()
    return action_space[action_index], action_index

def update_q_network(old_state, action_index, reward, new_state):
    old_state = torch.tensor(old_state, dtype=torch.float32)
    new_state = torch.tensor(new_state, dtype=torch.float32)
    reward = torch.tensor(reward, dtype=torch.float32)

    # Calcola la previsione corrente (Q(s, a))
    current_q = qnetwork(old_state)[action_index]

    # Calcola l'obiettivo (r + γ max Q(s', a'))
    with torch.no_grad():
        max_new_q = torch.max(qnetwork(new_state)).item()
    target_q = reward + 0.95 * max_new_q  # 0.95 è il fattore di sconto γ

    # Calcola la perdita
    loss = torch.square(target_q - current_q)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def main():
    global quadcopter_pos, quadcopter_path, q_table, action_space

    pygame.init()
    display = (1024*1.3,768*1.3)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    # Initial camera position
    glTranslatef(0.0, -1.0, -5)

    target_position = [np.random.uniform(-2, 2) for _ in range(3)]
    print(target_position)

    epsilon = 0.3  # Probabilità di scelta casuale dell'azione
    episodes = 1000  # Numero di episodi per l'apprendimento

     # Creare e ottimizzare la rete neurale
    qnetwork = QNetwork(state_size, action_size)
    optimizer = optim.Adam(qnetwork.parameters(), lr=0.001)

  
    # Carica i pesi del modello se esistono
    if os.path.exists('qnetwork_weights_episode.pth'):
        qnetwork.load_state_dict(torch.load('qnetwork_weights_episode.pth'))

    for episode in range(episodes):
        done = False
        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    exit()
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resizing here if necessary
                    pass

            action, action_index = choose_action(quadcopter_pos, epsilon)

            old_quadcopter_pos = quadcopter_pos[:]
            move_quadcopter(action_index)
            r = reward(old_quadcopter_pos, quadcopter_pos, target_position)
            update_q_network(old_quadcopter_pos, action_index, r, quadcopter_pos)
            quadcopter_path.append(quadcopter_pos[:])

            if is_colliding(quadcopter_pos, target_position):
                print("Episodio {} completato!".format(episode))
                print("Il drone ha colliso con l'obiettivo!")
                # Imposta il font e le dimensioni del testo
             
                # Stampa il numero di episodi completati
                print(f'Episodio {episode} completato!')
                # Stampa la posizione finale del drone
                print(f'Posizione finale: {quadcopter_pos}')
                # Stampa la posizione finale dell'obiettivo
                print(f'Posizione obiettivo: {target_position}')
                # Stampa il percorso del drone
                #print(f'Percorso: {quadcopter_path}')
                # Stampa la tabella Q
                #print(f'Tabella Q: {q_table}')
             
                
                done = True
                # Genera una nuova posizione target casualmente nel range -5 a 5 per ogni dimensione
                target_position = [np.random.uniform(-2, 2) for _ in range(3)]

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            Quadricottero()
            draw_target(target_position, 0.2, (1,0,0)) 
            # Save the current matrix
            


            # Reset perspective projection
            pygame.display.flip()
            pygame.time.wait(10)

        # Salva i pesi del modello dopo ogni episodio
        torch.save(qnetwork.state_dict(), f'qnetwork_weights_episode.pth')

main()

