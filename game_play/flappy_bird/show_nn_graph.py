from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, preprocess_obs
from stable_baselines3.common.utils import get_device, is_vectorized_observation
import gym_flappy_bird
import gym
import torch
import numpy as np
import os

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

if __name__ == "__main__":
    model = DQN.load(os.path.join(os.path.dirname(__file__), 'logs/best_model.zip'))
    print(model.policy.q_net)
    model.policy.q_net.features_extractor.cnn.register_forward_hook(get_activation('cnn'))
    model.policy.q_net.features_extractor.linear.register_forward_hook(get_activation('linear'))
    model.policy.q_net.q_net.register_forward_hook(get_activation('linear'))

    env = gym.make("flappy-bird-v0", is_demo=True)
    obs = env.reset()

    action, _ = model.predict(obs)

    print(activation['cnn'])