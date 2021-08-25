import numpy as np
import math
import random

def argmax(q_values):
    """
    Takes in a matrix of n*k q_values and returns the index
    of the item with the highest value for each row.
    Breaks ties randomly.
    returns: vector of size n, where each item is the index of
    the highest value in q_values for each row.
    """
    # Generate a mask of the max values for each row
    mask = q_values == q_values.max(axis=1)[:, None]
    # Generate noise to be added to the ties
    r_noise = 1e-6*np.random.random(q_values.shape)
    # Get the argmax of the noisy masked values
    return np.argmax(r_noise*mask, axis=1)


class GreedyAgent:
    def __init__(self, reward_estimates):
        """
        Our agent takes as input the initial reward estimates.
        This estimates will be updated incrementally after each
        interaction with the environment.
        """
        assert len(reward_estimates.shape) == 2

        self.num_bandits = reward_estimates.shape[1]
        self.num_experiments = reward_estimates.shape[0]
        self.reward_estimates = reward_estimates.astype(np.float64)
        self.action_count = np.zeros(reward_estimates.shape)

    def get_action(self):
        # Our agent is greedy, so there's no need for exploration.
        # Our argmax will do just fine for this situation
        action = argmax(self.reward_estimates)

        # Add a 1 to each action selected in the action count
        self.action_count[np.arange(self.num_experiments), action] += 1

        return action

    def update_estimates(self, reward, action):
        # rew is a matrix with the obtained rewards from our previous
        # action. Use this to update our estimates incrementally
        n = self.action_count[np.arange(self.num_experiments), action]

        # Compute the difference between the received rewards vs the reward estimates
        error = reward - self.reward_estimates[np.arange(self.num_experiments), action]

        # Update the reward difference incrementally
        self.reward_estimates[np.arange(self.num_experiments), action] += (1 / n) * error


def epsilon_greedy(epsilon):
    rand = np.random.random()
    if rand < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)

    return action


def softmax(tau):
    total = sum([math.exp(val / tau) for val in Q])
    probs = [math.exp(val / tau) / total for val in Q]

    threshold = random.random()
    cumulative_prob = 0.0
    for i in range(len(probs)):
        cumulative_prob += probs[i]
        if (cumulative_prob > threshold):
            return i
    return np.argmax(probs)

def UCB(iters):
    ucb = np.zeros(10)

    # explore all the arms
    if iters < 10:
        return i

    else:
        for arm in range(10):
            # calculate upper bound
            upper_bound = math.sqrt((2 * math.log(sum(count))) / count[arm])

            # add upper bound to the Q valyue
            ucb[arm] = Q[arm] + upper_bound

        # return the arm which has maximum value
        return (np.argmax(ucb))


def thompson_sampling(alpha, beta):
    samples = [np.random.beta(alpha[i] + 1, beta[i] + 1) for i in range(10)]

    return np.argmax(samples)