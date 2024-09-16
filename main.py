from experiment import run_experiment
from post_processing import plot_results

if __name__ == "__main__":
    # Hyperparameters common for both experiments
    k = 10  # Number of arms
    q_init = 0  # Initial q-values
    sigma = 1.0  # Variance of reward distribution
    mu_a = 0  # Mean of true action values
    sigma_a2 = 1.0  # Variance of true action values
    runs = 2000  # Number of independent runs
    timesteps = 1000  # Number of timesteps per run

    # Experiment 1 with epsilon = 0.1
    epsilon_1 = 0.1
    avg_rewards_1, opt_action_percent_1 = run_experiment(k=k, epsilon=epsilon_1, q_init=q_init, sigma=sigma, mu_action=mu_a, sigma_action=sigma_a2, runs=runs, timesteps=timesteps)

    # Experiment 2 with epsilon = 0.01
    epsilon_2 = 0.01
    avg_rewards_2, opt_action_percent_2 = run_experiment(k=k, epsilon=epsilon_2, q_init=q_init, sigma=sigma, mu_action=mu_a, sigma_action=sigma_a2, runs=runs, timesteps=timesteps)

    # Prepare the data for plotting
    experiments_data = [
        (avg_rewards_1, opt_action_percent_1, f"Epsilon = {epsilon_1}"),
        (avg_rewards_2, opt_action_percent_2, f"Epsilon = {epsilon_2}")
    ]

    # Plot the results for both experiments
    plot_results(experiments_data)