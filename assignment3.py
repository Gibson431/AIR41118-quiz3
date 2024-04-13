import sys
import gym
import pybullet as p
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from collections import defaultdict
import pickle
from IPython.display import clear_output
import torch
import random

import simple_driving
from simple_driving.envs.simple_driving_env import SimpleDrivingEnv


# display = Display(visible=0, size=(400, 300))
# display.start()
#
# def display_video(frames, framerate=30):
#   """Generates video from `frames`.
#
#   Args:
#     frames (ndarray): Array of shape (n_frames, height, width, 3).
#     framerate (int): Frame rate in units of Hz.
#
#   Returns:
#     Display object.
#   """
#   height, width, _ = frames[0].shape
#   dpi = 70
#   orig_backend = matplotlib.get_backend()
#   matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
#   fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
#   matplotlib.use(orig_backend)  # Switch back to the original backend.
#   ax.set_axis_off()
#   ax.set_aspect('equal')
#   ax.set_position([0, 0, 1, 1])
#   im = ax.imshow(frames[0])
#   def update(frame):
#     im.set_data(frame)
#     return [im]
#   interval = 1000/framerate
#   anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
#                                   interval=interval, blit=True, repeat=False)
#   return HTML(anim.to_html5_video())

# Hyper parameters that will be used in the DQN algorithm

EPISODES = 2500  # number of episodes to run the training for
MAX_EPISODE_LENGTH = 200
LEARNING_RATE = 0.00025  # the learning rate for optimising the neural network weights
MEM_SIZE = 50000  # maximum size of the replay memory - will start overwritting values once this is exceed
REPLAY_START_SIZE = 10000  # The amount of samples to fill the replay memory with before we start learning
BATCH_SIZE = 32  # Number of random samples from the replay memory we use for training each iteration
GAMMA = 0.99  # Discount factor
EPS_START = 0.1  # Initial epsilon value for epsilon greedy action sampling
EPS_END = 0.0001  # Final epsilon value
EPS_DECAY = 4 * MEM_SIZE  # Amount of samples we decay epsilon over
MEM_RETAIN = 0.1  # Percentage of initial samples in replay memory to keep - for catastrophic forgetting
NETWORK_UPDATE_ITERS = 5000  # Number of samples 'C' for slowly updating the target network \hat{Q}'s weights with the policy network Q's weights

FC1_DIMS = 128  # Number of neurons in our MLP's first hidden layer
FC2_DIMS = 128  # Number of neurons in our MLP's second hidden layer

# metrics for displaying training status
best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
np.bool = np.bool_

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# for creating the policy and target networks - same architecture
class Network(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n

        # build an MLP with 2 hidden layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(*self.input_shape, FC1_DIMS),  # input layer
            torch.nn.ReLU(),  # this is called an activation function
            torch.nn.Linear(FC1_DIMS, FC2_DIMS),  # hidden layer
            torch.nn.ReLU(),  # this is called an activation function
            torch.nn.Linear(FC2_DIMS, self.action_space),  # output layer
        )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()  # loss function

    def forward(self, x):
        return self.layers(x)


# handles the storing and retrival of sampled experiences
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros(
            (MEM_SIZE, *env.observation_space.shape), dtype=np.float32
        )
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros(
            (MEM_SIZE, *env.observation_space.shape), dtype=np.float32
        )
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
        # if memory count is higher than the max memory size then overwrite previous values
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            mem_index = int(
                self.mem_count % ((1 - MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE)
            )  # avoid catastrophic forgetting, retain first 10% of replay buffer

        self.states[mem_index] = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] = 1 - done

        self.mem_count += 1

    # returns random samples from the replay buffer, number is equal to BATCH_SIZE
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones = self.dones[batch_indices]

        return states, actions, rewards, states_, dones


class DQN_Solver:
    def __init__(self, env, device=None):
        self.device = device
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)  # Q
        self.target_network = Network(env)  # \hat{Q}
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )  # initially set weights of Q to \hat{Q}
        self.learn_count = (
            0  # keep track of the number of iterations we have learnt for
        )
        if self.device:
            self.policy_network.to(self.device)
            self.target_network.to(self.device)

    # epsilon greedy
    def choose_action(self, observation):
        # only start decaying epsilon once we actually start learning, i.e. once the replay memory has REPLAY_START_SIZE
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
                -1.0 * self.learn_count / EPS_DECAY
            )
        else:
            eps_threshold = 1.0
        # if we rolled a value lower than epsilon sample a random action
        if random.random() < eps_threshold:
            # return np.random.choice(np.array(range(9)))
            return np.random.choice(
                np.array(range(9)), p=[0.15, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            )

        # otherwise policy network, Q, chooses action with highest estimated Q-value so far
        state = torch.tensor(observation).float().detach()
        state = state.unsqueeze(0)
        self.policy_network.eval()  # only need forward pass
        with torch.no_grad():  # so we don't compute gradients - save memory and computation
            q_values = self.policy_network(state)
        return torch.argmax(q_values).item()

    # main training loop
    def learn(self):
        states, actions, rewards, states_, dones = (
            self.memory.sample()
        )  # retrieve random batch of samples from replay memory
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.policy_network.train(True)
        q_values = self.policy_network(
            states
        )  # get current q-value estimates (all actions) from policy network, Q
        q_values = q_values[batch_indices, actions]  # q values for sampled actions only

        self.target_network.eval()  # only need forward pass
        with torch.no_grad():  # so we don't compute gradients - save memory and computation
            q_values_next = self.target_network(
                states_
            )  # target q-values for states_ for all actions (target network, \hat{Q})

        q_values_next_max = torch.max(q_values_next, dim=1)[
            0
        ]  # max q values for next state

        q_target = rewards + GAMMA * q_values_next_max * dones  # our target q-value

        loss = self.policy_network.loss(
            q_values, q_target
        )  # compute loss between estimated q-values (policy network, Q) and target (target network, \hat{Q})
        # compute gradients and update policy network Q weights
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        # set target network \hat{Q}'s weights to policy network Q's weights every C steps
        if self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            print("updating target network")
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def returning_epsilon(self):
        return self.exploration_rate


def main():
    if len(sys.argv) > 1:
        env = None
        if sys.argv[1] == "test":
            env = SimpleDrivingEnv(renders=True)
            agent = Network(env)
            agent.load_state_dict(torch.load("policies.bak/policy_network.pkl"))

            try:
                agent.load_state_dict(torch.load("policies.bak/policy_network.pkl"))
            except:
                print("No existing model. Please train first")
                return
            state = env.reset()
            step = 0
            frames = []
            for i in range(200):
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)
                frames.append(
                    env.render()
                )  # if running locally not necessary unless you want to grab onboard camera image
                if done:
                    break
            # while True:
            #     # sampling loop - sample random actions and add them to the replay buffer
            #             # otherwise policy network, Q, chooses action with highest estimated Q-value so far
            #     state = torch.tensor(state).float().detach()
            #     state = state.unsqueeze(0)
            #     agent.eval()  # only need forward pass
            #     with torch.no_grad():       # so we don't compute gradients - save memory and computation
            #         q_values = agent(state)
            #     action = torch.argmax(q_values).item()
            #     state_, reward, done, info = env.step(action)
            #     frames.append(env.render())
            #     state = state_

            #     if done:
            #         break
            #     step += 1
        else:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            env = SimpleDrivingEnv()
            # set manual seeds so we get same behaviour everytime - so that when you change your hyper parameters you can attribute the effect to those changes
            env.action_space.seed(0)
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            episode_batch_score = 0
            episode_reward = 0
            agent = None
            if use_cuda:
                agent = DQN_Solver(env, device)  # create DQN agent
            else:
                agent = DQN_Solver(env)  # create DQN agent

            if sys.argv[1] == "retrain":
                try:
                    agent.policy_network.load_state_dict(
                        torch.load("policies.bak/policy_network.pkl")
                    )
                    agent.target_network.load_state_dict(
                        agent.policy_network.state_dict()
                    )
                    epsilon_start = 0.2
                except:
                    print("No existing model. Please train first")
                    env.close()
                    return
            elif sys.argv[1] != "train":
                print(f"Unknown option: {sys.argv[1]}")

            for i in range(EPISODES):
                state = (
                    env.reset()
                )  # this needs to be called once at the start before sending any actions
                while True:
                    # sampling loop - sample random actions and add them to the replay buffer
                    action = agent.choose_action(state)
                    state_, reward, done, info = env.step(action)
                    agent.memory.add(state, action, reward, state_, done)

                    # only start learning once replay memory reaches REPLAY_START_SIZE
                    if agent.memory.mem_count > REPLAY_START_SIZE:
                        agent.learn()

                    state = state_
                    episode_batch_score += reward
                    episode_reward += reward

                    if done:
                        break

                episode_history.append(i)
                episode_reward_history.append(episode_reward)
                episode_reward = 0.0

                # save our model every batches of 100 episodes so we can load later. (note: you can interrupt the training any time and load the latest saved model when testing)
                if i % 500 == 0 and agent.memory.mem_count > REPLAY_START_SIZE:
                    torch.save(
                        agent.policy_network.state_dict(),
                        f"policies.bak/policy_network_e{i}.pkl",
                    )
                    print(
                        "average total reward per episode batch since episode ",
                        i,
                        ": ",
                        episode_batch_score / float(100),
                    )
                    episode_batch_score = 0
                elif agent.memory.mem_count < REPLAY_START_SIZE:
                    if i % 10 == 0:
                        print(
                            f"waiting for buffer to fill. {agent.memory.mem_count}/{REPLAY_START_SIZE}"
                        )
                    episode_batch_score = 0

            torch.save(
                agent.policy_network.state_dict(), f"policies.bak/policy_network.pkl"
            )

            plt.plot(episode_history, episode_reward_history)
            plt.show()
        env.close()


if __name__ == "__main__":
    main()
