import gymnasium as gym
import numpy as np
import random

# Create Frozen Lake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Q-Learning parameters
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))
alpha = 0.8 #Learning rate
gamma = 0.95 #Discount factor
epsilon = 0.2 #Exploration rate
episodes = 5000
max_steps = 100

# Training
for ep in range(episodes):
    state, info = env.reset()
    for step in range(max_steps):
        # ε-greedy action selection
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        if done:
            break

# Test the agent
state, info = env.reset()
path = [state]
done = False
while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    path.append(state)

print("Learned path:", path)
