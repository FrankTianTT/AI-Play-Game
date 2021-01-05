from stable_baselines3 import DQN
import gym_flappy_bird
from utility import CnnEvalCallback
from utility import CustomCNN
import gym


env = gym.make("flappy-bird-v0")
eval_env = gym.make("flappy-bird-v0")

eval_callback = CnnEvalCallback(eval_env=eval_env,
                             eval_freq=int(3e3),
                             log_path="logs",
                             best_model_save_path="logs")

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = DQN(policy="CnnPolicy",
            env=env,
            batch_size=32,
            buffer_size=5000,
            learning_starts=250,
            policy_kwargs=policy_kwargs,
            tensorboard_log="log")

print(model.policy)

if __name__ == "__main__":
    model.learn(int(1e7),
                callback=eval_callback)