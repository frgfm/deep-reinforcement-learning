import random
import numpy as np
from collections import defaultdict
random.seed(a=17)

class Agent:

    def __init__(self, nA=6, eps=5e-3, alpha=1., gamma=1.):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.eps: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return random.choice(np.arange(self.nA))

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
        current = self.Q[state][action]  # estimate in Q-table (for current state, action pair)
        # get value of state, action pair at next time step
        Qsa_max = 0
        if next_state is not None:
            prob = np.ones(self.nA) * self.eps / self.nA  # current policy (for next state S')
            prob[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)
            Qsa_max = np.dot(self.Q[next_state], prob) 
        target = reward + (self.gamma * Qsa_max)               # construct TD target
        self.Q[state][action] = current + (self.alpha * (target - current)) # get updated value

        if done:
            self.eps /= 2
