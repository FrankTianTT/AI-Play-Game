from stable_baselines3 import DQN
import gym_flappy_bird
import gym
import os


env = gym.make("flappy-bird-v0")
obs = env.reset()

model = DQN.load(os.path.join(os.path.dirname(__file__), 'logs/best_model.zip'))

print(model.policy)
# if __name__ == "__main__":
#     rewards = 0
#     time_steps = 0
#     while True:
#         # action = env.action_space.sample()
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         rewards += reward
#         time_steps += 1
#         env.render()
#
#         if done:
#             obs = env.reset()
#             print("rewards: {}, of {} steps".format(rewards, time_steps))
#             rewards = 0
#             time_steps = 0