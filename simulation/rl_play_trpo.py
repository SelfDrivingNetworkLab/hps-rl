import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import time

from Steed.drllib import model4trpo, trpo, utils


import torch
import torch.optim as optim
import torch.nn.functional as F


RunName = "Test7"

BESTMODEL = "testingtrials/aktr4/best_-4667031.914_0.dat"#"saves/d4pg-Test6/best_-49936.050_2452730.dat" #best_-4488.278_700780.dat" #best_-124.831_1751950.dat"

def pred_net(net, env, device="cpu"):
    
    ##MK to add experience buffer class
    buffer = utils.ExperienceBuffer(env.obs_names, env.action_names)
    rewards = 0.0
  
    steps = 0
    obs = env.reset()
    while True:
        obs_v = utils.float32_preprocessor([obs]).to(device)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.cpu().numpy()
        action = np.clip(action, -1, 1)

        obs, reward, done,  divers = env.step(action)
       

        action_scaled = env.scale_action(action)
        obs_scaled = env.scale_obs(obs)
        buffer.append(action_scaled,obs_scaled,reward,e_cost) #,net_energy,net_power)

        rewards += reward
       
        steps += 1
        if done:
            break
    actions_df = buffer.action_data()
    obs_df = buffer.obs_data()
    reward_df = buffer.reward_data()

    actions_df.to_csv('preds/aktr4/actions_df.csv',index=False)
    obs_df.to_csv('preds/aktr4/obs_df.csv',index=False)
    reward_df.to_csv('preds/aktr4/reward_df.csv',index=False)
    #net_energy_df.to_csv('preds/net_energy_df_ddpg_multienv_002.csv',index=False)
    #net_power_df.to_csv('preds/net_power_df_ddpg_multienv_002.csv',index=False)

    return rewards, steps

def run_trpo(actiontype,env,device):

    if actiontype == "Discrete":
        if env.action_space.n == 2:
            act_net = model4trpo.ModelActor(env.observation_space.shape[0], env.action_space.n - 1).to(device)
        else:
            act_net = model4trpo.ModelActor(env.observation_space.shape[0], env.action_space.n).to(device)
    else:
        act_net = model4trpo.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    print(act_net)

    best_model = torch.load(BESTMODEL)
    act_net.load_state_dict(best_model)
    print("1")
    act_net.train(False)
    act_net.eval()

    frame_idx = 0
    best_reward = None

    ts = time.time()
    rewards, steps, e_costs = pred_net(act_net, env, device=device)
    print("Test done in %.2f sec, reward %.3f, e_cost %.3f, steps %d" % (time.time() - ts, rewards, e_costs, steps))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--actiontype", required=True, help='Discrete or Continuous')
    parser.add_argument("--env", required=True, help="Env")
    parser.add_argument("--testenv", required=True, help="Test Env")
    parser.add_argument("--cuda", required=True, default=False, help="Cuda")
    args = parser.parse_args()

    action_type = args["actiontype"]
    env = args["env"]
    cuda = args["cuda"]
    device = torch.device("cuda" if cuda else "cpu")

    run_trpo(action_type,env,device)