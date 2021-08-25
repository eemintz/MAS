import numpy as np
from bandits import BanditTenArmedGaussian


# means = np.array([[5, 1, 0, -10]]) # The mean for a four-armed bandit. Single experiment
# stdev = np.array([[1, 0.1, 5, 1]]) # The standard deviation for a four-armed bandit.

env = BanditTenArmedGaussian()
# env.seed(5)

# number of rounds (iterations)
num_rounds = 20000

# Count of number of times an arm was pulled
count = np.zeros(10)

# Sum of rewards of each arm
sum_rewards = np.zeros(10)

# Q value which is the average reward
Q = np.zeros(10)

for i in range(num_rounds):
    # Select the arm using UCB
    arm = UCB(i)

    # Get the reward
    observation, reward, done, info = env.step(arm)

    # update the count of that arm
    count[arm] += 1

    # Sum the rewards obtained from the arm
    sum_rewards[arm] += reward

    # calculate Q value which is the average rewards of the arm
    Q[arm] = sum_rewards[arm] / count[arm]

print('The optimal arm is {}'.format(np.argmax(Q)))

for i in range(10):
    next_state, reward, done, _ = env.step(0)
    print(reward)




