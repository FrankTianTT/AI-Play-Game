from stable_baselines3 import DQN
import gym_flappy_bird
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
import torch
import numpy as np
import os
import pathlib

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def draw(out3d, time_steps, save_path):
    fig = plt.figure(facecolor='black')
    ax = Axes3D(fig)
    ax.set_facecolor((0, 0, 0))
    plt.axis('off')

    size = out3d.shape
    draw_list = []
    for i in range(3):
        list = []
        for n in range(size[i]):
            if n % 3 == 0:
                list.append(n)
        draw_list.append(list)

    x_dot = []
    y_dot = []
    z_dot = []
    for i in draw_list[0]:
        for j in draw_list[1]:
            for k in draw_list[2]:
                if out3d[i][j][k] > 0:
                    z_dot.append(i)
                    x_dot.append(j)
                    y_dot.append(k)

    ax.scatter3D(x_dot, y_dot, z_dot, marker='.', c='b', s=5)  # 绘制散点图
    plt.savefig(os.path.join(save_path, '{}.jpg'.format(time_steps)))
    plt.close(fig)

model = DQN.load(os.path.join(os.path.dirname(__file__), 'logs/best_model.zip'))

model.policy.q_net.features_extractor.relu1.register_forward_hook(
    get_activation('relu1'))
model.policy.q_net.features_extractor.relu1.register_forward_hook(
    get_activation('relu2'))
model.policy.q_net.features_extractor.relu1.register_forward_hook(
    get_activation('relu3'))
model.policy.q_net.features_extractor.relu1.register_forward_hook(
    get_activation('relu4'))
model.policy.q_net.features_extractor.relu1.register_forward_hook(
    get_activation('relu5'))

nn_save_path = 'nn_graph'
pathlib.Path(nn_save_path).mkdir(parents=True, exist_ok=True)

env = gym.make("flappy-bird-v0", is_demo=True)
obs = env.reset()

if __name__ == "__main__":
    rewards = 0
    time_steps = 0
    rollout = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        draw(activation['relu4'][0], time_steps, nn_save_path)
        obs, reward, done, info = env.step(action)
        env.render()

        rewards += reward
        time_steps += 1

        if done:
            obs = env.reset()
            print("rewards: {}, of {} steps".format(rewards, time_steps))
            rewards = 0
            time_steps = 0
            rollout += 1
