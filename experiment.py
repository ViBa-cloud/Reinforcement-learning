from agent import EpsilonGreedyAgent
import numpy as np

def run_experiment(k=10, epsilon=0.1, q_init=0, sigma=1.0, mu_action=0, sigma_action=1.0, runs=2000, timesteps=1000):
    """
    Run the k-armed bandit experiment and return the average rewards and % optimal actions.
    :param k: Number of arms
    :param epsilon: Exploration probability
    :param q_init: Initial value estimates
    :param sigma: Reward variance for sampling
    :param mu_action: Mean for arm distributions
    :param sigma_action: Variance for arm distributions
    :param runs: Number of independent runs
    :param timesteps: Number of timesteps per run
    :return: Average reward and % optimal action over all runs
    """
    average_rewards = np.zeros(timesteps)
    optimal_action_counts = np.zeros(timesteps)

    for run in range(runs):
        agent = EpsilonGreedyAgent(k, epsilon, q_init, sigma, mu_action, sigma_action)
        rewards = np.zeros(timesteps)
        optimal_actions = np.zeros(timesteps)

        for t in range(timesteps):
            action = agent.select_action()
            reward = agent.get_reward(action)
            agent.q_value_update(action, reward)

            rewards[t] = reward
            if action == agent.optimal_action:
                optimal_actions[t] = 1

        average_rewards += rewards
        optimal_action_counts += optimal_actions

    # Averaging results across runs
    average_rewards /= runs
    optimal_action_percentage = (optimal_action_counts / runs) * 100

    return average_rewards, optimal_action_percentage