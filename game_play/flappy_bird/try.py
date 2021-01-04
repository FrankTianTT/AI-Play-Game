import stable_baselines3
import gym_flappy_bird
import gym

env = gym.make("flappy-bird-v0")
obs = env.reset()

print(obs.shape)