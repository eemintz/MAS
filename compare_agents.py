import argparse
import numpy as np
import matplotlib.pyplot as plt
from bandits import BanditTenArmedGaussian
from agents import UcbAgent, EpsilonGreedyAgent

# settings
NUM_ACTIONS = 10

# number of iterations
NUM_TRIALS = 1000
NUM_STEPS = 1000

AGENT_EPSILON = "epsilon"
AGENT_UCB = "ucb"


def run_epsilon_experiment(epsilon):
    env = BanditTenArmedGaussian()
    agent = EpsilonGreedyAgent(epsilon=epsilon)

    # initialize reward
    reward = 0.0
    rewards = np.zeros((NUM_TRIALS, NUM_STEPS + 1))
    averages = np.zeros(NUM_STEPS)

    for j in range(NUM_TRIALS):
        for i in range(NUM_STEPS):
            # Select the arm using according agent policy
            arm = agent.get_action(reward)

            # Get the reward
            observation, reward, done, info = env.step(arm)

            # Store the average cumulative score
            rewards[j, i + 1] = rewards[j, i] + reward

        agent.reset()

    for i in range(1, NUM_STEPS):
        averages[i] = np.mean(rewards[:, i + 1] / (i + 1))

    return averages


def run_ucb_experiment():
    env = BanditTenArmedGaussian()
    agent = UcbAgent()

    # initialize reward
    reward = 0.0
    rewards = np.zeros((NUM_TRIALS, NUM_STEPS + 1))
    averages = np.zeros(NUM_STEPS)

    for j in range(NUM_TRIALS):
        for i in range(NUM_STEPS):
            # Select the arm using according agent policy
            arm = agent.get_action(reward)

            # Get the reward
            observation, reward, done, info = env.step(arm)

            # Store the average cumulative score
            rewards[j, i + 1] = rewards[j, i] + reward

        agent.reset()

    for i in range(NUM_STEPS):
        averages[i] = np.mean(rewards[:, i + 1] / (i + 1))

    return averages


if __name__ == "__main__":
    average_01 = run_epsilon_experiment(epsilon=0.1)
    average_04 = run_epsilon_experiment(epsilon=0.4)
    average_07 = run_epsilon_experiment(epsilon=0.7)

    # plot average rewards
    plt.figure()
    plt.plot(average_01, label='e = 0.1')
    plt.plot(average_04, label='e = 0.4')
    plt.plot(average_07, label='e = 0.7')
    plt.xticks([0, 250, 500, 750, 1000])
    plt.yticks([0.0, 0.5, 1.0, 1.5])
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.title("Average reward for different epsilon")

    plt.legend()

    plt.savefig("Egreedy")
    plt.show()

    average_ucb = run_ucb_experiment()
    # plot average rewards
    plt.figure()
    plt.plot(average_ucb, label='UCB')
    plt.xticks([0, 250, 500, 750, 1000])
    plt.yticks([0.0, 0.5, 1.0, 1.5])
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.title("Average reward")

    plt.legend()

    plt.savefig("UCB")
    plt.show()
