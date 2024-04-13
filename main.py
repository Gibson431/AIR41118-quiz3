import gym
from simple_driving.envs.simple_driving_env import SimpleDrivingEnv
from assignment3 import Network

# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random

env = SimpleDrivingEnv(renders=True)
policy_network = Network(env)
policy_network.load_state_dict(torch.load("policies.bak/policy_network.pkl"))
# frames = []
# frames.append(env.render())
for _ in range(5):
    state = env.reset()
    for i in range(200):
        state = torch.tensor(state).float().detach()
        state = state.unsqueeze(0)
        policy_network.eval()  # only need forward pass
        with torch.no_grad():  # so we don't compute gradients - save memory and computation
            q_values = policy_network(state)
        action = torch.argmax(q_values).item()

        state, reward, done, _ = env.step(action)
        # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
        if done:
            break

env.close()
