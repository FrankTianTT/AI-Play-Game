# by frank tian, 2021-1-16

from stable_baselines3 import DQN
import gym_flappy_bird
from stable_baselines3.common.callbacks import EvalCallback
import gym

env = gym.make("FlappyBirdFeature-v1")
eval_env = gym.make("FlappyBirdFeature-v1")

eval_callback = EvalCallback(eval_env=eval_env,
                             eval_freq=5000,
                             log_path="logs",
                             best_model_save_path="logs")

model = DQN(policy="MlpPolicy",
            env=env,
            batch_size=32,
            buffer_size=1000000,
            learning_starts=50000,
            tensorboard_log="log")

print(model.policy)

if __name__ == "__main__":
    model.learn(int(1e7),
                callback=eval_callback)