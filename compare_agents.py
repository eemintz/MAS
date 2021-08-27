import argparse
import numpy as np
import matplotlib.pyplot as plt
from bandits import BanditTenArmedGaussian
from agents import UcbAgent, EpsilonGreedyAgent

# settings
NUM_ACTIONS = 10

NUM_TRIALS = 2000
NUM_STEPS = 1000

AGENT_EPSILON = "epsilon"
AGENT_UCB = "ucb"


def main():

    # setup algorithms and arrays to hold results
    env = BanditTenArmedGaussian()
    agents = {AGENT_EPSILON: EpsilonGreedyAgent(), AGENT_UCB: UcbAgent()}

    rewards = {key: np.zeros(NUM_STEPS, dtype=np.float32) for key in agents.keys()}
    optimal_actions = {key: np.zeros(NUM_STEPS, dtype=np.float32) for key in agents.keys()}

    # run experiment
    for i in range(NUM_TRIALS):

        for j in range(NUM_STEPS):

            for agent in agents.values():
                agent.act()

        optimal_action = np.argmax(env.action_values)

        for key in agents.keys():
            rewards[key] += agents[key].rewards
            optimal_actions[key] += np.array(agents[key].actions) == optimal_action
            agents[key].reset()

        env.reset()

    # average rewards and optimal actions
    for key in agents.keys():
        rewards[key] = rewards[key] / NUM_TRIALS
        optimal_actions[key] = (optimal_actions[key] / NUM_TRIALS) * 100

    # plot average rewards
    for i, key in enumerate(sorted(rewards.keys())):
        plt.plot(rewards[key], label=key)

    plt.xticks([0, 250, 500, 750, 1000])
    plt.yticks([0.0, 0.5, 1.0, 1.5])
    plt.xlabel("Steps")
    plt.ylabel("Average reward")

    # if args.title is not None:
    #     plt.title(args.title)

    plt.legend()

    # plt.savefig("{:s}_rewards.{:s}".format(args.save_path, args.format))
    plt.show()

    # plot optimal actions
    for key in sorted(optimal_actions.keys()):
        plt.plot(optimal_actions[key], label=key)

    plt.xticks([0, 250, 500, 750, 1000])
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.xlabel("Steps")
    plt.ylabel("Optimal action (%)")

    # if args.title is not None:
    #     plt.title(args.title)

    plt.legend()

    # plt.savefig("{:s}_actions.{:s}".format(args.save_path, args.format))
    plt.show()


if __name__ == "__main__":
    main()
