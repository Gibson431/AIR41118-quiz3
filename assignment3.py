import gym
import simple_driving
import sys
import pybullet as p
import numpy as np
import math
from collections import defaultdict, deque
import pickle
import torch
import torch.optim as optim
import random

from agent import DeepQAgent

from simple_driving.envs.simple_driving_env import SimpleDrivingEnv


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def q_learning(
    env,
    agent,
    episodes=1000,
    max_episode_length=200,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    batch_size=32,
    buffer_size=10000,
    target_update=10,
    device=torch.device("cpu"),
):
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(buffer_size)
    target_agent = DeepQAgent(
        input_size=agent.input_size,
        num_classes=agent.num_classes,
    )
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()

    agent.to(device)
    target_agent.to(device)

    epsilon = epsilon_start

    for episode in range(episodes):
        if episode % 1000 == 0:
            agent.save_model("agent-bak.pkl")
        state = env.reset()
        total_reward = 0
        for t in range(max_episode_length):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_values = agent(state_tensor)
                    action = torch.argmax(action_values, dim=1).item()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                current_q_values = (
                    agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                )
                next_q_values = target_agent(next_states).max(1)[0].detach()
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = torch.nn.functional.mse_loss(current_q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if episode % target_update == 0:
            target_agent.load_state_dict(agent.state_dict())
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return agent


def main():
    max_episode_length = 100  # maximum number of steps for episode roll out

    if len(sys.argv) > 1:
        env = None
        if sys.argv[1] == "test":
            env = SimpleDrivingEnv(renders=True)
            agent = DeepQAgent(2, env.action_space.n)
            try:
                agent.load_model("agent.pkl")
            except:
                print("No existing model. Please train first")
                return

            for _ in range(5):
                state = env.reset()
                steps = 0
                while steps < max_episode_length:
                    # in case policy gets stuck (shouldn't happen if valid path exists and optimal policy learnt)

                    ########## policy is simply taking the action with the highest Q-value for given state ##########
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action_values = agent(state_tensor)
                        action = torch.argmax(action_values, dim=1).item()

                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    # frames.append(env.render(mode="rgb_array"))
                    steps += 1
                    if done:
                        break

        else:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            env = SimpleDrivingEnv()
            agent = DeepQAgent(2, env.action_space.n)

            episodes = 10000
            max_episode_length = max_episode_length
            gamma = 0.95
            epsilon_start = 1.0
            epsilon_end = 0.05
            epsilon_decay = 0.99
            batch_size = 100
            buffer_size = 10000
            target_update = 30
            if sys.argv[1] == "retrain":
                try:
                    agent.load_model("agent.pkl")
                    epsilon_start = 0.2
                except:
                    print("No existing model. Please train first")
                    env.close()
                    return

            env.action_space.seed(0)  # so we get same the random sequence every time
            state = env.reset()
            agent = q_learning(
                env,
                agent,
                episodes=episodes,
                max_episode_length=max_episode_length,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                batch_size=batch_size,
                buffer_size=buffer_size,
                target_update=target_update,
                device=device,
            )
            agent.save_model("agent.pkl")

        env.close()


if __name__ == "__main__":
    main()
