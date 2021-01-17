from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent(nA=env.action_space.n, eps=1e-2, alpha=0.3, gamma=0.9)
avg_rewards, best_avg_reward = interact(env, agent)
