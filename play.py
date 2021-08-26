import numpy as np
from bandits import BanditTenArmedGaussian
from agents import UcbAgent


env = BanditTenArmedGaussian()
# env.seed(5)

agent = UcbAgent()

# number of rounds (iterations)
num_rounds = 20000
# initialize reward
reward = 0.0

for i in range(num_rounds):
    # Select the arm using UCB
    arm = agent.get_action(reward)

    # Get the reward
    observation, reward, done, info = env.step(arm)

print('The optimal arm is {}'.format(np.argmax(agent.get_qvalues())))
print(agent.get_qvalues())


