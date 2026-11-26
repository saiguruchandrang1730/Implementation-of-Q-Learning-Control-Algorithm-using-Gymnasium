# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
The objective of this project is to implement the Q-Learning algorithm using the Windy Gridworld environment provided by the gym-classics2 package.

Windy Gridworld is a grid-based environment in which an agent must navigate from a starting position to a goal position. The environment contains wind in certain columns, which pushes the agent upward when it moves. Therefore, the agent must learn to select suitable actions while considering the effect of wind.

The agent interacts with the environment, receives a reward of -1 for each step, and updates its Q-table using the Q-Learning algorithm. The main objective is to learn an effective policy that helps the agent reach the goal with minimum steps and cumulative penalty.

## Software Requirements
- Python 3.x
- Google Colab or Jupyter Notebook
- Gymnasium
- NumPy
- Matplotlib


## Environment Description

Windy Gridworld is a 7 × 10 grid environment consisting of 70 displayed grid positions.

The agent starts at position (0, 3) and must reach the goal at position (7, 3), using the environment's coordinate convention.

The environment contains different wind strengths in different columns. When the agent moves, the wind affects its vertical position and may push it upward.

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
1.Import the required libraries, including Gymnasium, gym-classics2, NumPy, and Matplotlib.

2.Create the Windy Gridworld environment using gym.make().

3.Obtain the number of states and actions from the environment.

4.Initialize the Q-table with zeros.

5.Set the learning rate, discount factor, epsilon value, and number of training episodes.

6.Reset the environment at the beginning of each episode.

7.Select an action using the epsilon-greedy strategy.

8.Execute the selected action and observe the next state and reward.

9.Update the Q-table using the Q-Learning update equation.

10.Repeat the process until the goal is reached or the maximum number of steps is completed.

11.Reduce epsilon gradually to balance exploration and exploitation.

12.After training, calculate the state-value function and extract the learned policy.

13.Display the final Q-table, estimated state-value function, learned policy, and learning curve.


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
[[-13.755 -13.671 -13.875 -13.483]
 [-15.047 -14.32  -14.217 -14.454]
 [-15.148 -15.    -15.085 -15.   ]
 [-16.    -15.    -16.    -16.   ]
 [-15.31  -15.    -15.756 -15.649]
 [-14.91  -14.747 -14.729 -14.974]
 [-14.779 -14.456 -14.983 -14.5  ]
 [-13.127 -13.094 -13.399 -13.138]
 [-13.972 -13.531 -13.754 -14.017]
 [-14.784 -14.    -14.233 -15.55 ]
 [-15.    -14.    -15.    -16.   ]
 [-14.596 -14.    -14.711 -15.229]
 [-14.135 -13.948 -14.146 -14.672]
 [-14.    -13.813 -14.045 -14.319]
 [-12.678 -12.267 -12.552 -13.461]
 [-13.153 -12.72  -12.642 -13.397]
 [-13.675 -13.    -13.063 -14.947]
 [-14.    -13.    -14.    -15.   ]
 [-13.331 -13.    -13.907 -14.884]
 [-13.382 -12.998 -13.286 -13.061]
 [-13.    -12.991 -13.262 -13.585]
 [-12.268 -11.434 -11.444 -12.174]
 [-11.95  -11.876 -11.829 -12.716]
 [-12.784 -12.    -12.75  -13.373]
 [-13.    -12.    -13.    -14.   ]
 [-12.71  -12.    -12.816 -13.502]
 [-12.361 -12.    -12.937 -13.514]
 [-12.845 -12.    -12.089 -12.203]
 [-10.684 -10.619 -11.    -11.692]
 [-11.252 -10.965 -11.    -11.434]
 [-11.641 -11.    -11.5   -12.532]
 [-12.    -11.    -12.    -13.   ]
 [-11.931 -11.    -11.996 -12.78 ]
 [-11.75  -11.    -11.969 -12.938]
 [-10.326  -9.793  -9.808 -10.115]
 [-10.306  -9.995 -10.    -10.417]
 [-10.921 -10.    -10.922 -11.921]
 [-11.    -10.    -11.    -12.   ]
 [-10.984 -10.    -10.996 -11.998]
 [ -9.307  -8.951  -9.16   -9.888]
 [ -9.224  -9.     -9.25   -9.176]
 [ -9.734  -9.     -9.547  -9.424]
 [-10.     -9.    -10.    -11.   ]
 [ -2.23   -2.216  -1.467  -4.906]
 [ -5.737  -2.138  -1.     -6.974]
 [  0.      0.      0.      0.   ]
 [ -7.437  -6.152  -6.789  -6.73 ]
 [ -8.026  -8.     -8.238  -8.536]
 [ -9.     -8.     -9.    -10.   ]
 [ -1.719  -2.413  -2.461  -1.438]
 [ -2.188  -2.102  -2.5    -2.   ]
 [ -7.     -4.     -2.     -1.   ]
 [ -6.851  -5.     -5.     -5.928]
 [ -7.252  -6.     -6.716  -8.754]
 [ -7.845  -7.     -7.744  -8.795]
 [ -8.     -7.     -8.     -9.   ]
 [ -2.973  -2.5    -2.5    -2.327]
 [ -3.     -3.     -3.084  -3.   ]
 [ -4.     -3.     -4.     -2.   ]
 [ -5.     -4.     -3.     -6.   ]
 [ -6.     -5.     -4.     -7.   ]
 [ -7.     -6.     -5.     -8.   ]
 [ -7.     -7.     -6.     -8.   ]]

Estimated State-Value Function:
[[-14.46 -13.81 -12.99 -12.   -11.   -10.    -9.    -8.    -7.    -6.  ]
 [-14.73 -13.95 -13.   -12.   -11.   -10.    -9.    -8.    -7.    -5.  ]
 [-15.   -14.   -13.   -12.   -11.   -10.    -9.    -6.15  -6.    -4.  ]
 [-15.   -14.   -13.   -12.   -11.    -9.99  -8.95   0.    -5.    -3.  ]
 [-15.   -14.   -13.   -12.   -10.96  -9.79   0.    -1.    -1.    -2.  ]
 [-14.22 -13.53 -12.64 -11.83 -10.62   0.     0.    -1.47  -2.    -3.  ]
 [-13.48 -13.09 -12.27 -11.43   0.     0.     0.     0.    -1.44  -2.33]]

Learned Policy:
[['R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'D']
 ['D' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'D']
 ['R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'D']
 ['S' 'R' 'R' 'R' 'R' 'R' 'R' 'G' 'R' 'D']
 ['R' 'R' 'R' 'R' 'R' 'R' '' 'D' 'L' 'L']
 ['D' 'R' 'D' 'D' 'R' '' '' 'D' 'L' 'L']
 ['L' 'R' 'R' 'R' '' '' '' '' 'L' 'L']]

Average reward over last 1000 episodes: -15.98
```

---
<img width="847" height="437" alt="image" src="https://github.com/user-attachments/assets/83058bd6-d994-4589-8808-8c1b80f418c6" />

## Result

```text
The Q-Learning control algorithm was successfully implemented using the Windy Gridworld environment in Gymnasium.

The agent learned to navigate through the grid while considering the effect of wind and move towards the goal state.

The training produced the following results:

Final Q-table containing the learned action values.

Estimated state-value function representing the maximum Q-value for each state.

Learned policy showing the preferred action in each grid position.

Learning curve illustrating the change in average reward during training.

Average reward over the last 1000 episodes: -15.98.

The learned policy shows the agent selecting actions that guide it from the starting position towards the goal.


```

---

## Inference

```text

From the implementation, it is observed that the Q-Learning algorithm enables the agent to learn suitable actions through repeated interaction with the Windy Gridworld environment.

Initially, the agent explores the environment by selecting random actions. As training progresses, it gradually learns which actions provide better future rewards.

The negative Q-values occur because the agent receives a reward of -1 for every step. Actions that lead to reaching the goal in fewer steps generally have higher Q-values (less negative values).

The learned policy and state-value function demonstrate that the agent has learned to navigate the windy environment and move towards the goal.

Therefore, Q-Learning is an effective model-free reinforcement learning algorithm for learning action-selection policies in the Windy Gridworld environment.

```

---

