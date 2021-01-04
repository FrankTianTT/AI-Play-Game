import stable_baselines3
import gym_flappy_bird
import gym

env = gym.make("flappy-bird-v0", is_demo=True)
obs = env.reset()


if __name__ == "__main__":
    print(env.observation_space)
    print(obs.shape)
    # rewards = 0
    # time_steps = 0
    # while True:
    #     action = 0 # env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     rewards += reward
    #     time_steps += 1
    #     env.render()
    #
    #     if done:
    #         obs = env.reset()
    #         print("rewards: {}, of {} steps".format(rewards, time_steps))
    #         rewards = 0
    #         time_steps = 0