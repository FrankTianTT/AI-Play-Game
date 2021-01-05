from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
import torch
import gym
import torch.nn as nn
import os
import gym_flappy_bird

class CnnEvalCallback(EvalCallback):
    """
    this is a EvalCallback for CnnPolicy, which is for dealing with the "type error" of training-env and eval-env.
    """
    def __init__(self,
                eval_env,
                callback_on_new_best=None,
                n_eval_episodes: int = 5,
                eval_freq: int = 10000,
                log_path: str = None,
                best_model_save_path: str = None,
                deterministic: bool = True,
                render: bool = False,
                verbose: int = 1):
        super().__init__(eval_env=eval_env,
                         callback_on_new_best=callback_on_new_best,
                         n_eval_episodes=n_eval_episodes,
                         eval_freq=eval_freq,
                         log_path=log_path,
                         best_model_save_path=best_model_save_path,
                         deterministic=deterministic,
                         render=render,
                         verbose=verbose)
        self.eval_env = VecTransposeImage(self.eval_env)

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.Conv2d1 = nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0)
        self.relu1 = nn.ReLU()
        self.Conv2d2 = nn.Conv2d(16, 32, kernel_size=8, stride=4, padding=0)
        self.relu2 = nn.ReLU()
        self.Conv2d3 = nn.Conv2d(32, 32, kernel_size=6, stride=3, padding=0)
        self.relu3 = nn.ReLU()
        self.Conv2d4 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0)
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():

            n_flatten = self.forward_cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear1 = nn.Linear(n_flatten, features_dim)
        self.relu5 = nn.ReLU()

    def forward_cnn(self, x):
        x = self.Conv2d1(x)
        x = self.relu1(x)
        x = self.Conv2d2(x)
        x = self.relu2(x)
        x = self.Conv2d3(x)
        x = self.relu3(x)
        x = self.Conv2d4(x)
        x = self.relu4(x)
        x = self.flatten(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_cnn(x)
        x = self.linear1(x)
        x = self.relu5(x)
        return x


if __name__ == "__main__":
    # env = gym.make("flappy-bird-v0")
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=128),
    # )
    # model = DQN(policy="CnnPolicy", env=env,)
    #
    # model_customized = DQN(policy="CnnPolicy", env=env, policy_kwargs=policy_kwargs)
    #
    # print(model_customized.policy)
    #
    # total_params = sum(p.numel() for p in model.policy.parameters())
    # total_trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    # print('model:\ntotal parameters: {}, training parameters: {}'.format(total_params, total_trainable_params))
    # # total parameters: 125984068, training parameters: 125984068
    #
    # total_params = sum(p.numel() for p in model_customized.policy.parameters())
    # total_trainable_params = sum(p.numel() for p in model_customized.policy.parameters() if p.requires_grad)
    # print('customized model:\ntotal parameters: {}, training parameters: {}'.format(total_params, total_trainable_params))
    # # total parameters: 203748, training parameters: 203748

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    env = gym.make("flappy-bird-v0")

    model = DQN.load(os.path.join(os.path.dirname(__file__), 'logs/best_model.zip'))

    model.policy.q_net.features_extractor.Conv2d1.register_forward_hook(
        get_activation('Conv2d1'))
    model.policy.q_net.features_extractor.relu2.register_forward_hook(
        get_activation('relu2'))
    model.policy.q_net.features_extractor.relu3.register_forward_hook(
        get_activation('relu3'))
    model.policy.q_net.features_extractor.relu4.register_forward_hook(
        get_activation('relu4'))
    model.policy.q_net.features_extractor.relu5.register_forward_hook(
        get_activation('relu5'))

    obs = env.reset()

    action, _ = model.predict(obs)

    print(activation['Conv2d1'])