#!/usr/bin/env python3
from gymlabs.envs import cartpole_env

import os

import time

import argparse
import numpy as np

from drllib import models



import torch

CUDA = False
BESTMODEL = "saves_LRC/saves/DRL_Master/Test_203/best_-69177.359_7533385.dat" #replace wil file name
RunName = "Test5"

def pred_net(net, env, device="cpu"):
    

    #MK: add experience buffer Ask Melis
    #buffer = cartpole_env.ExperienceBuffer(env.obs_names, env.action_names)

    obs = env.reset()
    rewards = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, _ = env.step(action)
        rewards += reward
        steps += 1
        if done:
            break

    return steps, rewards

def run_ddpg(actiontype,env,device):

    save_path = os.path.join("saves", "ddpg-" + RunName)
    os.makedirs(save_path, exist_ok=True)
    if actiontype == "Discrete":
        if env.action_space.n == 2:
            net = models.DDPGActor(env.observation_space.shape[0], env.action_space.n).to(device)
        else:
            net = models.DDPGActor(env.observation_space.shape[0], env.action_space.n).to(
                device)
    else:
        net = models.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(
            device)
    best_model = torch.load(BESTMODEL)
    net.load_state_dict(best_model)
    net.train(False)
    net.eval()

    frame_idx = 0
    best_reward = None
    ts = time.time()
    rewards, steps = pred_net(net, env, device=device)

    print("In %d steps we got %.3f reward" % (steps, rewards))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--actiontype", required=True, help='Discrete or Continuous')
    parser.add_argument("--env", required=True, help="Env")
    parser.add_argument("--cuda",required=True,help="Cuda")
    args = parser.parse_args()

    action_type = args["actiontype"]
    env = args["env"]
    cuda = args["cuda"]

    device = torch.device("cuda" if cuda else "cpu")

    run_ddpg(action_type,env,device)


