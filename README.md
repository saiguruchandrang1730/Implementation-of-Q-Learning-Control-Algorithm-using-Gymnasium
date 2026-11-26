# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement


## Software Requirements
- Python 3.x
- Google Colab or Jupyter Notebook
- Gymnasium
- NumPy
- Matplotlib


## Environment Description



## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm



## Python Program

```python

!pip install gym-classics2

import gymnasium as gym
import gym_classics2
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make(
    "WindyGridworld-v0",
    tabular=True,
    step_reward=-1.0,
    goal_reward=-1.0
)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of States:", n_states)
print("Number of Actions:", n_actions)

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def choose_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])
    # -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset(seed=42 if episode == 0 else None)

    total_reward = 0

    for step in range(max_steps_per_episode):

        action = choose_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += alpha * (target - Q[state, action])

        state = next_state

        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)


state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)

print("Training completed!")


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")

    value_grid = np.zeros((7, 10))

    for state in range(n_states):

        x, y = env.unwrapped.id2state(state)

        value_grid[y, x] = values[state]

    print(np.round(np.flipud(value_grid), 2))


def print_policy(policy):

    action_symbols = {
        0: "U",
        1: "R",
        2: "D",
        3: "L"
    }

    policy_grid = np.empty((7, 10), dtype="<U1")

    for state in range(n_states):

        x, y = env.unwrapped.id2state(state)

        policy_grid[y, x] = action_symbols[policy[state]]

    policy_grid[3, 0] = "S"
    policy_grid[3, 7] = "G"

    print("\nLearned Policy:")
    print(np.flipud(policy_grid))

# -------------------------------------------------
# Final Q-table and Results
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:",
      round(average_reward, 2))

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 100

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title("Q-Learning Learning Curve - Windy Gridworld")

plt.grid(True)

plt.show()

env.close()



```
---

## Output

```text
Final Q-table:





Estimated State-Value Function:






Learned Policy:




Average reward over last 1000 episodes: 
```

---

## Result

```text



```

---

## Inference

```text



```

---

