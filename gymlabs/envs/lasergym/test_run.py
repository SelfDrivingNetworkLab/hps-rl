import argparse
import logging

from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from Steed.gymlabs.envs.lasergym.mimo_env import  LaserControllerEnv
import gym


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-m", type=int, default=3, choices=[3, 9], help="MxM beam shape")
    p.add_argument("-p", "--perturb", type=float, default=180, help="start phase perturb range [deg]")
    p.add_argument("-t", "--threshold", type=float, default=0.6, help="target norm. efficiency")
    args = p.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    config = {"M": args.m, "done_threshold": args.threshold, "perturb_range": args.perturb}
    env = LaserControllerEnv()

    check_env(env)
    print("Env check passed.")

    print("Instantiate the agent...")
    model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./runs/")
    print("Train the agent...")
    model.learn(total_timesteps=10000)

    print("Save the agent...")
    model.save("saves/ddpg_laser")
    # del model  # delete trained model to demonstrate loading
    print("Load the saved agent...")
    model = DDPG.load("saves/ddpg_laser")

    print("Evaluate the agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    print("Testing agent...")
    obs = env.reset()
    for ix in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

        