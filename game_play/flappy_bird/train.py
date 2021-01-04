from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
import gym_flappy_bird
import gym

class CnnEvalCallback(EvalCallback):
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

env = gym.make("flappy-bird-v0")
eval_env = gym.make("flappy-bird-v0")

eval_callback = CnnEvalCallback(eval_env=eval_env,
                             eval_freq=int(1e4),
                             log_path="logs",
                             best_model_save_path="logs")

model = DQN(policy="CnnPolicy",
            env=env,
            buffer_size=50000,
            learning_starts=2500,
            tensorboard_log="log")

if __name__ == "__main__":
    model.learn(int(1e5),
                callback=eval_callback)