"""
File holds self contained TRPO agent.
"""

import torch
import torch.nn as nn
import gym
from numpy.random import choice
from copy import deepcopy
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters


class DeepQAgent(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepQAgent, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Layer 1
        out = self.fc1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        # Layer 2
        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        # Layer 3
        out = self.fc3(out)
        # out = self.bn3(out)
        out = self.relu(out)

        # Layer 4
        out = self.fc4(out)
        # out = self.bn4(out)
        out = self.relu(out)

        # Output
        out = nn.functional.softmax(out, dim=1)

        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)
