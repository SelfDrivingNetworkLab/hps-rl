#!/usr/bin/env python3
import argparse
from Steed.drllib import models
import numpy as np
import torch

CUDA = False
BESTMODEL = "saves_LRC/saves/DRL_Master/Test_203/best_-69177.359_7533385.dat" #replace wil file name
RunName = "Test5"

def run_a2c(actiontype,env,device):

    if actiontype == "Discrete":
        if env.action_space.n == 2:
            net = models.ModelA2C(env.observation_space.shape[0], env.action_space.n - 1).to(device)
        else:
            net = models.ModelA2C(env.observation_space.shape[0], env.action_space.n).to(device)
    else:
        net = models.ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    net.load_state_dict(torch.load(BESTMODEL))

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v, var_v, val_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actiontype", required=True, help='Discrete or Continuous')
    parser.add_argument("--env", required=True, help="Env")
    parser.add_argument("--cuda", required=True, help="Cuda")
    args = parser.parse_args()

    action_type = args["actiontype"]
    env = args["env"]
    cuda = args["cuda"]
    device = torch.device("cuda" if cuda else "cpu")

    # spec = gym.envs.registry.spec(args.env)
    # spec._kwargs['render'] = False
    # env = gym.make(args.env)
    # if args.record:
    #     env = gym.wrappers.Monitor(env, args.record)
    run_a2c(action_type,env,device)
