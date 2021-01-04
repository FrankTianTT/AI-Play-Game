from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import gym_flappy_bird
import gym

env = gym.make("flappy-bird-v0")
eval_env = gym.make("flappy-bird-v0")

eval_callback = EvalCallback(eval_env=eval_env,
                             eval_freq=int(1e4),
                             log_path="logs",
                             best_model_save_path="logs")

model = DQN(policy="CnnPolicy",
            env=env,
            tensorboard_log="log")

if __name__ == "__main__":
    model.learn(int(1e5))