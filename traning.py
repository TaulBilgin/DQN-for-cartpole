import torch
import gymnasium as gym
import numpy as np
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dqn_input(state, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity):
    Cart_Velocity_dqn = np.digitize(state[1], Cart_Velocity)
    Pole_Angle_dqn = np.digitize(state[2], Pole_Angle)
    Pole_Angular_Velocity_dqn = np.digitize(state[3], Pole_Angular_Velocity)
    
    input_tensor = torch.tensor([Cart_Velocity_dqn, Pole_Angle_dqn, Pole_Angular_Velocity_dqn], dtype=torch.float32).to(device)
    
    return input_tensor

def optimize(memory, policy_dqn, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity, learning_rate, gamma):
    # last 10 step of memory
    lest_memory = memory[-10:]
    
    # Initialize the optimizer and loss function
    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    current_q_list = []
    target_q_list = []

    # Iterate over each transition in the sampled batch
    for now_state, action, new_state, reward, done in lest_memory:
        # Assign a high reward if the episode is finished successfully
        if done:
            target = -10
        else:
            with torch.no_grad(): 
                target = reward + gamma * policy_dqn(dqn_input(new_state, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity)).max().item()
        # Get the current Q-value
        current_q = policy_dqn(dqn_input(now_state, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity))
        current_q_list.append(current_q)
        
        
        # Create a copy of the current Q-values for updating
        target_q = current_q.clone()
        target_q[action] = target # Update the Q-value for the taken action
        target_q_list.append(target_q)
    
    # Compute the loss between current and target Q-values
    loss = loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
    
# check the average step of dqn
def test_for_save(policy_dqn, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity):
    env2 = gym.make('CartPole-v1')
    totel_step = 0

    for i in range(10):
        now_state = env2.reset()[0]  # Reset environment and get initial state
        done = False  # Flag to check if the episode is finished
        step = 0

        # Play one episode
        while not done and step < 10000 :
            with torch.no_grad():
                action = policy_dqn(dqn_input(now_state, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity)).argmax().item()  # Best action
            step += 1
            totel_step += 1
            # Take action and observe result
            new_state, reward, done, truncated, _ = env2.step(action)
            
            now_state = new_state

    return (totel_step / 10)
        
    

def train (past_best_save_input):

    env = gym.make('CartPole-v1')
    	
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
    learning_rate = 0.01
    gamma = 0.9


    past_best_save = past_best_save_input
    memory = []
    run = 0
    save_count = 0


    for _ in range(10000):
        now_state = env.reset()[0]  # Reset environment and get initial state
        done = False  # Flag to check if the episode is finished
        step = 0
        run += 1  # Increment the episode counte

        # Play one episode
        while not done and step < 10000 :
            with torch.no_grad():
                action = policy_dqn(dqn_input(now_state, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity)).argmax().item()  # Best action
            step += 1
            # Take action and observe result
            new_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in memory
            memory.append((now_state, action, new_state, reward, done))
            
            now_state = new_state

        print(run)
        
        # save the dqn that highest step 
        if step > 1000:
            env.close()
            best_save = test_for_save(policy_dqn, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity)
            if best_save > past_best_save:
                save_count += 1
                past_best_save = best_save
                torch.save(policy_dqn.state_dict(), f"CartPole{best_save}.pt")
            
        optimize(memory, policy_dqn, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity, learning_rate, gamma)
        memory = []

    return save_count
        
while True:
    save_count = train(500) # average minimum step for saving
    if save_count != 0 :
        break