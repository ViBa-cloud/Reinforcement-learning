import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, k, epsilon, q_init, sigma, mu_action, sigma_action ):
        """
        Initialize the k-armed bandit agent with ε-greedy strategy.
        :param k: Number of arms
        :param epsilon: Exploration probability
        :param q_init: Initial value estimates
        :param sigma: Reward variance for sampling
        :param mu_action: Mean for arm distributions
        :param sigma_action: Variance for arm distributions
        """
        self.k = k
        self.epsilon = epsilon
        self.q_init = q_init
        self.sigma = sigma
        self.q_values = np.full(k,q_init)
        self.action_ct = np.zeros(k)
        self.q_star = np.random.normal(mu_action, np.sqrt(sigma_action),k)
        self.optimal_action = np.argmax(self.q_star) 

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)  # explore
        else:
            return np.argmax(self.q_values)  # exploit
        
    def q_value_update(self, action, reward):
        self.action_ct[action] += 1
        self.q_values[action] += (1.0 / self.action_ct[action]) * (reward - self.q_values[action])

    def get_reward(self, action):
        return np.random.normal(self.q_star[action], self.sigma)
    
    

        
