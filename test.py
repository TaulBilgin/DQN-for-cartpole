import gymnasium as gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dqn_input(state, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity):
    Cart_Velocity_dqn = np.digitize(state[1], Cart_Velocity)
    Pole_Angle_dqn = np.digitize(state[2], Pole_Angle)
    Pole_Angular_Velocity_dqn = np.digitize(state[3], Pole_Angular_Velocity)
    
    input_tensor = torch.tensor([Cart_Velocity_dqn, Pole_Angle_dqn, Pole_Angular_Velocity_dqn], dtype=torch.float32).to(device)

    return input_tensor

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, h3_nodes, out_actions):
        super(DQN, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.fc3 = nn.Linear(h2_nodes, h3_nodes)
        self.out = nn.Linear(h3_nodes, out_actions)

    def forward(self, x):
        # Define the forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


env = gym.make('CartPole-v1', render_mode="human")

# Divide position and velocity into segments
Cart_Velocity = np.linspace(-3.40, 3.40, 100) # Between -3.40 and 3.40
Pole_Angle = np.linspace(-0.418, 0.418, 100)  # Between -0.418 ana 0.418
Pole_Angular_Velocity = np.linspace(-3.40, 3.40, 100) # Between -3.40 and 3.40
"""
The code doesn't work like : Cart_Velocity = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 100)
you must put manual value of env.observation_space.low[0] and env.observation_space.high[0]
"""
in_states = 3
h1_nodes = 128
h2_nodes = 128
h3_nodes = 128
out_actions = 2
policy_dqn = DQN(in_states, h1_nodes, h2_nodes, h3_nodes, out_actions).to(device)

# Load the trained model weights
policy_dqn.load_state_dict(torch.load("your model name"))

# Switch the model to evaluation mode
policy_dqn.eval()

run = 0   

for _ in range(100) :
    now_state = env.reset()[0]  # Reset environment and get initial stat
    done = False  # Flag to check if the episode is finished
    step = 0
    run += 1  # Increment the episode counte

    # Play one episode
    while not done and step < 10000 :
        # Use the policy network to select the best action
        with torch.no_grad():
            action = policy_dqn(dqn_input(now_state, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity)).argmax().item()  # Best action
        step += 1
        
        # Take action and observe result
        new_state, reward, done, truncated, _ = env.step(action)
        # Store transition in memory
        now_state = new_state
        
    print(step)