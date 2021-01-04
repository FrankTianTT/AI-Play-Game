from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
import torch
import gym
import torch.nn as nn
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
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

if __name__ == "__main__":
    env = gym.make("flappy-bird-v0")
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = DQN(policy="CnnPolicy", env=env,)

    model_customized = DQN(policy="CnnPolicy", env=env, policy_kwargs=policy_kwargs)

    total_params = sum(p.numel() for p in model.policy.parameters())
    total_trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print('model:\ntotal parameters: {}, training parameters: {}'.format(total_params, total_trainable_params))
    # total parameters: 125984068, training parameters: 125984068

    total_params = sum(p.numel() for p in model_customized.policy.parameters())
    total_trainable_params = sum(p.numel() for p in model_customized.policy.parameters() if p.requires_grad)
    print('customized model:\ntotal parameters: {}, training parameters: {}'.format(total_params, total_trainable_params))
    # total parameters: 734404, training parameters: 734404
