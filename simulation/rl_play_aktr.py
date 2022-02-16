import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import time
import argparse
from Steed.drllib import model4trpo, trpo, utils


import torch
import torch.optim as optim
import torch.nn.functional as F


RunName = "Test7"

BESTMODEL = "testingtrials/aktr6/best_-2102615.436_2803120.dat"#"saves/d4pg-Test6/best_-49936.050_2452730.dat" #best_-4488.278_700780.dat" #best_-124.831_1751950.dat"

def pred_net(net, env, device="cpu"):
    
    buffer = env.ExperienceBuffer(env.obs_names, env.action_names)
    rewards = 0.0
    e_costs = 0.0
    steps = 0
    obs = env.reset()
    while True:
        obs_v = utils.float32_preprocessor([obs]).to(device)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.cpu().numpy()
        action = np.clip(action, -1, 1)

        obs, reward, done,  divers = env.step(action)
        e_cost = divers[0]
        # net_energy = divers[1]
        # net_power = divers[2]

        action_scaled = env.scale_action(action)
        obs_scaled = env.scale_obs(obs)
        buffer.append(action_scaled,obs_scaled,reward,e_cost) #,net_energy,net_power)

        rewards += reward
        e_costs += e_cost
        steps += 1
        if done:
            break
    actions_df = buffer.action_data()
    obs_df = buffer.obs_data()
    reward_df = buffer.reward_data()
    e_costs_df = buffer.e_cost_data()
    #net_energy_df = buffer.net_energy_data()
    #net_power_df = buffer.net_power_data()
    actions_df.to_csv('preds/aktr6/actions_df.csv',index=False)
    obs_df.to_csv('preds/aktr6/obs_df.csv',index=False)
    reward_df.to_csv('preds/aktr6/reward_df.csv',index=False)
    e_costs_df.to_csv('preds/aktr6/e_costs_df.csv',index=False)
    #net_energy_df.to_csv('preds/net_energy_df_ddpg_multienv_002.csv',index=False)
    #net_power_df.to_csv('preds/net_power_df_ddpg_multienv_002.csv',index=False)

    return rewards, steps, e_costs

def run_aktr(actiontype,env,device):
    if actiontype == "Discrete":
        if env.action_space.n == 2:
            act_net = model4trpo.ModelActor(env.observation_space.shape[0], env.action_space.n - 1).to(device)
        else:
            act_net = model4trpo.ModelActor(env.observation_space.shape[0], env.action_space.n).to(device)
    else:
        act_net = model4trpo.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    # net_act = tpro_model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)

    best_model = torch.load(BESTMODEL)
    print("1")

    act_net.load_state_dict(best_model, strict=False)
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
    run_aktr(action_type,env,device)




