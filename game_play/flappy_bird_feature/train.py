# by frank tian, 2021-1-14

from stable_baselines3 import DQN
import gym_flappy_bird
from stable_baselines3.common.callbacks import EvalCallback
import gym

env = gym.make("FlappyBirdFeature-v0")
eval_env = gym.make("FlappyBirdFeature-v0")

eval_callback = EvalCallback(eval_env=eval_env,
                             eval_freq=int(3e3),
                             log_path="logs",
                             best_model_save_path="logs")

model = DQN(policy="MlpPolicy",
            env=env,
            batch_size=32,
            buffer_size=5000,
            learning_starts=250,
            tensorboard_log="log")

print(model.policy)

if __name__ == "__main__":
    model.learn(int(1e7),
                callback=eval_callback)