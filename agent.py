import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nS=500, nA=6):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.nS = nS
        # initialize action-value function (empty dictionary of arrays)
        #self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.Q = np.zeros((self.nS, self.nA))
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01
        
        self.alpha = 0.1
        self.gamma = 0.8
        
    def select_action(self, env, state, i_episode):
        # print("States are {}".format(dict(self.Q)))
        #action_probabilities = self.epsilon_greedy_probs(env, self.Q[state], i_episode, max(self.epsilon*self.epsilon_decay, self.epsilon))
        #action = np.random.choice(self.nA, p=action_probabilities)
        self.epsilon *= self.epsilon_decay
        if random.uniform(0,1) > self.epsilon:
            action = np.argmax(self.Q[state,:])
        else:
            action = env.action_space.sample()
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #self.Q[state][action] += 1
        #if not done:
        self.Q[state][action] += (self.alpha * (reward + self.gamma * np.max(self.Q[next_state,:]) - self.Q[state][action]))
        #print(self.Q[state][action])
        
        #action_probabilities = self.epsilon_greedy_probs(env, self.Q[state], i_episode, self.epsilon)  
    
    def epsilon_greedy_probs(self, env, Q_s, i_episode, epsilon):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(env.env.nA) * epsilon / env.env.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.env.nA)
        return policy_s
    
