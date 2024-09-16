import matplotlib.pyplot as plt

def plot_results(experiments_data):
    """
    Plot the results of multiple experiments for comparison.
    
    :param experiments_data: A list of tuples containing (average_rewards, optimal_action_percentage, label)
                             for each experiment.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot average rewards
    for avg_rewards, _, label in experiments_data:
        ax1.plot(avg_rewards, label=f"Average Reward - {label}")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Average Reward")
    ax1.legend()

    # Plot % optimal action selected
    for _, opt_action_percent, label in experiments_data:
        ax2.plot(opt_action_percent, label=f"% Optimal Action - {label}")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("% Optimal Action")
    ax2.legend()

    plt.tight_layout()
    plt.show()
