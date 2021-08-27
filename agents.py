import numpy as np
import math
import random


class EpsilonGreedyAgent:
    def __init__(self, arms=10, epsilon=0.1):
        self.last_action = None
        self.action_values = np.zeros(arms)
        self.action_count = np.zeros(arms)
        self.sum_rewards = np.zeros(arms)
        self.epsilon = epsilon

    def __update_action_values(self, action, reward):
        self.action_count[action] += 1
        step_size = 1 / self.action_count[action]  # default behavior is averaging samples
        self.action_values[action] += step_size * (reward - self.action_values[action])
        self.sum_rewards[action] += reward  # sum the rewards obtained from the arm

    def get_action(self, reward):
        should_explore = np.random.random() < self.epsilon

        if should_explore:
            current_action = np.random.randint(0, len(self.action_count))
        else:
            current_action = np.argmax(self.action_values)

        if self.last_action is not None:
            self.__update_action_values(self.last_action, reward)
        self.last_action = current_action
        return current_action

    def get_action_values(self):
        return self.action_values

    def reset(self):
        self.last_action = None
        self.action_values *= 0.0
        self.action_count *= 0.0
        self.sum_rewards *= 0.0


class UcbAgent:

    def __init__(self, arms=10):
        self.last_action = None
        self.q_values = np.zeros(arms)
        self.sum_rewards = np.zeros(arms)
        self.action_count = np.zeros(arms)

    def get_action_values(self):
        return self.q_values

    def get_action(self, reward):
        ucb = np.zeros(10)

        if self.last_action is not None:
            self.action_count[self.last_action] += 1  # update the count of that arm
            self.sum_rewards[self.last_action] += reward  # sum the rewards obtained from the arm
            # calculate Q value which is the average rewards of the arm
            self.q_values[self.last_action] += (1 / self.action_count[self.last_action]) * (reward - self.q_values[self.last_action])
            #  self.q_values[self.last_action] = self.sum_rewards[self.last_action] / self.action_count[self.last_action]

        # explore all the arms
        untried_action = self.__find_untried_action()
        if untried_action is not None:
            current_action = untried_action
        else:
            for arm in range(10):
                # calculate upper bound
                upper_bound = math.sqrt((2 * math.log(sum(self.action_count))) / self.action_count[arm])

                # add upper bound to the Q value
                ucb[arm] = self.q_values[arm] + upper_bound

            # return the arm which has maximum value
            current_action = np.argmax(ucb)

        self.last_action = current_action
        return current_action

    def __find_untried_action(self):
        untried_actions = np.where(self.action_count == 0)[0]
        if len(untried_actions) == 0:
            return None
        return np.random.choice(untried_actions)

    def reset(self):
        self.last_action = None
        self.action_count *= 0.0
        self.q_values *= 0.0
        self.sum_rewards *= 0.0



