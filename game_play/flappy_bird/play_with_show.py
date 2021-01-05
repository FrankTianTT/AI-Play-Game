# by frank tian, 2021-1-15

from stable_baselines3 import DQN
import gym_flappy_bird
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
import torch
import numpy as np
import os
import pathlib
import cv2

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def draw3D(out3d, time_steps, save_path):
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
    s = []
    max = 0
    min = 0
    for i in draw_list[0]:
        for j in draw_list[1]:
            for k in draw_list[2]:
                if out3d[i][j][k] > 0:
                    s.append(out3d[i][j][k])
                    z_dot.append(i)
                    x_dot.append(j)
                    y_dot.append(k)
                    if out3d[i][j][k] > max:
                        max = out3d[i][j][k]
                    if out3d[i][j][k] < min:
                        min = out3d[i][j][k]
    s = np.array(s)
    s = (s - min) / (max - min) * 20
    ax.scatter3D(x_dot, y_dot, z_dot, marker='.', c='white', s=s)  # 绘制散点图
    plt.savefig(os.path.join(save_path, '{}.jpg'.format(time_steps)))
    plt.close(fig)

def draw1D(out1d, time_steps, save_path, drop=10):
    fig = plt.figure(facecolor='black')
    ax = Axes3D(fig)
    ax.set_facecolor((0, 0, 0))
    plt.axis('off')

    zd = []
    min = 0
    max = 0
    s = []
    for i in range(len(out1d)):
        if out1d[i] > 0:
            zd.append(i / drop)
            s.append(out1d[i])
            if out1d[i] > max:
                max = out1d[i]
            if out1d[i] < min:
                min = out1d[i]
    s = np.array(s)
    s = (s - min) / (max - min) * 40
    z = np.linspace(0, len(out1d) / drop, 1000)
    x = 5 * np.sin(z)
    y = 5 * np.cos(z)
    xd = 5 * np.sin(zd)
    yd = 5 * np.cos(zd)
    ax.scatter3D(xd, yd, zd, s=s, c='white')
    ax.plot3D(x, y, z, 'gray')
    plt.savefig(os.path.join(save_path, '{}.jpg'.format(time_steps)))
    plt.close(fig)

def save_obs(obs, time_steps, save_path):
    img = np.transpose(obs, (1, 0, 2))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(time_steps)), img)

model = DQN.load(os.path.join(os.path.dirname(__file__), 'logs/best_model.zip'))

model.policy.q_net.features_extractor.relu1.register_forward_hook(
    get_activation('relu1'))
model.policy.q_net.features_extractor.relu2.register_forward_hook(
    get_activation('relu2'))
model.policy.q_net.features_extractor.relu3.register_forward_hook(
    get_activation('relu3'))
model.policy.q_net.features_extractor.relu4.register_forward_hook(
    get_activation('relu4'))
model.policy.q_net.features_extractor.relu5.register_forward_hook(
    get_activation('relu5'))
model.policy.q_net.q_net.register_forward_hook(
    get_activation('q_net'))

save_path = 'show_save'

env = gym.make("flappy-bird-v0", is_demo=True)
obs = env.reset()

if __name__ == "__main__":
    rewards = 0
    time_steps = 0
    rollout = 0
    while True:
        this_rollout_save_path = os.path.join(save_path, str(rollout))

        action, _ = model.predict(obs, deterministic=True)
        nn_save_path = os.path.join(this_rollout_save_path, "nn_graph")

        # save nn graph
        for name in ['relu1', 'relu2', 'relu3', 'relu4']:
            layer_save_path = os.path.join(nn_save_path, name)
            pathlib.Path(layer_save_path).mkdir(parents=True, exist_ok=True)
            draw3D(activation[name][0], time_steps, layer_save_path)

        for name in ['relu5']:
            layer_save_path = os.path.join(nn_save_path, name)
            pathlib.Path(layer_save_path).mkdir(parents=True, exist_ok=True)
            draw1D(activation[name][0], time_steps, layer_save_path)

        image_save_path = os.path.join(this_rollout_save_path, "image")
        pathlib.Path(image_save_path).mkdir(parents=True, exist_ok=True)

        # save render image
        save_obs(obs, time_steps, image_save_path)


        obs, reward, done, info = env.step(action)
        env.render()

        rewards += reward
        time_steps += 1

        if done:
            obs = env.reset()
            print("rollout: {}, rewards: {}, steps: {}".format(rollout, rewards, time_steps))
            rewards = 0
            time_steps = 0
            rollout += 1
