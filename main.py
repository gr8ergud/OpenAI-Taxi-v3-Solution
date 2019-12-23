from agent import Agent
from monitor import interact
import gym
import numpy as np

#env = gym.make('Taxi-v2')
env = gym.make('Taxi-v3')
#env.render()

agent = Agent()
#print(env.action_space)
#print(env.observation_space)
#print(dir(env))

avg_rewards, best_avg_reward = interact(env, agent)