# droneSimRL
This project is a drone simulator that uses reinforcement learning for the drone's movement. It's built using Python, with Pygame and PyOpenGL for the graphical user interface and PyTorch for the machine learning component.

The key elements of the project are:

The Graphical Interface: Pygame is used to create a game window and handle user inputs. PyOpenGL, which provides Python bindings to the OpenGL library for rendering 2D and 3D graphics, is used to draw the drone and its environment.

The Drone Simulator: The drone, referred to as a "quadcopter" in the code, is represented by a series of components, each of which is modeled as a 3D cube with a specific size, color, and position. The function move_quadcopter changes the drone's position based on the selected action.

The Reinforcement Learning: The project employs Q-Learning, a type of reinforcement learning, where an agent learns to perform actions in an environment to maximize a cumulative reward. The QNetwork, defined in the code, is a neural network used to approximate the Q-value function, which gives the expected reward value of a certain action in a given state.

Choosing Actions: The function choose_action picks the action for the drone, which could be random (exploration), or it could be the action that maximizes the expected reward value as determined by the QNetwork (exploitation).

Updating The QNetwork: The function update_q_network updates the weights of the QNetwork based on the difference between the expected reward and the observed reward, using the Adam optimizer.

Main Function: The main function controls the main game loop. It creates a game window, generates a random target, and then in an infinite loop, chooses an action for the drone, moves the drone, calculates the reward, updates the QNetwork, and redraws the drone and its environment.

To install the dependencies from the `requirements.txt` file and run the Python program `drone_sim_v2_3d.py`, follow these steps:

1. **Open the terminal** (or command prompt, if you are using Windows).

2. **Navigate to your project's directory** using the `cd` command. For example, if your project is in a directory called "my_project" on the desktop, you would type `cd Desktop/my_project`.

3. **Install the dependencies** listed in the `requirements.txt` file. This can be done using the pip command `pip install -r requirements.txt`. This command tells pip to install all of the libraries listed in the `requirements.txt` file. If you are using a virtual environment, make sure it's activated before you run this command. If you're not using a virtual environment, you might need to use `pip3` instead of `pip`, or `python -m pip` if your Python installation is not in your system's PATH.

4. **Run your Python program**. Once all dependencies are installed, you can run your Python program using the command `python drone_sim_v2_3d.py`. Again, if your Python installation is not in your system's PATH, you might need to use `python3` instead of `python`.

Here's what this would look like:

cd Desktop/my_project
pip install -r requirements.txt
python drone_sim_v2_3d.py

Remember that these commands should be executed in the terminal/command prompt, and make sure to replace "Desktop/my_project" with the actual path to your project.

https://www.youtube.com/watch?v=z8D8lx083Kc
