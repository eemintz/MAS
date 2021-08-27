import numpy as np
from bandits import BanditTenArmedGaussian
from agents import UcbAgent, EpsilonGreedyAgent


env = BanditTenArmedGaussian()
# agent = UcbAgent()
agent = EpsilonGreedyAgent(epsilon=0.1)

# number of rounds (iterations)
num_rounds = 1000
# initialize reward
reward = 0.0

for i in range(num_rounds):
    # Select the arm using according agent policy
    arm = agent.get_action(reward)

    # Get the reward
    observation, reward, done, info = env.step(arm)

optimal_arm = np.argmax(agent.get_action_values())
print(f'The optimal arm is {optimal_arm}')
print(agent.get_action_values().round(2))


