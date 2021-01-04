from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import gym
import torch.nn as nn

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
