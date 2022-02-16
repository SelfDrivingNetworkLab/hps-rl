#!/usr/bin/env python3
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from Steed.optimizationfns import MultiClassLM
from Steed.drllib import models, utils, common

TEST_ITERS = 1000
RunName = "Test5"


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = utils.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def run_ac2(actiontype,env,test_env,device,OPT_FUNC,GAMMA,BATCH,LR,ENTROPY_BETA,STEP, REWARD_STEPS):

    save_path = os.path.join("saves", "ddpg-" + RunName)
    os.makedirs(save_path, exist_ok=True)
    if actiontype == "Discrete":
        if env.action_space.n == 2:
            net = models.ModelA2C(env.observation_space.shape[0], env.action_space.n - 1).to(device)
        else:
            net = models.ModelA2C(env.observation_space.shape[0], env.action_space.n).to(device)
    else:
        net = models.ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net)
    agent = models.AgentA2C(net, device=device)
    exp_source = utils.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    batch = []
    best_reward = None
    for step_idx, exp in enumerate(exp_source):
        rewards_steps = exp_source.pop_rewards_steps()
        if rewards_steps:
            rewards, steps = zip(*rewards_steps)

        if step_idx % TEST_ITERS == 0:
            ts = time.time()
            rewards, steps = test_net(net, test_env, device=device)
            print("Test done is %.2f sec, reward %.3f, steps %d" % (
                time.time() - ts, rewards, steps))
            if best_reward is None or best_reward < rewards:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                    fname = os.path.join(save_path, name)
                    torch.save(net.state_dict(), fname)
                best_reward = rewards

        batch.append(exp)
        if len(batch) < BATCH:
            continue

        states_v, actions_v, vals_ref_v = \
            common.unpack_batch_a2c(batch, net, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
        batch.clear()
        if  OPT_FUNC is None or OPT_FUNC == "adam":
            optimizer = optim.Adam(net.parameters(), lr=LR)
            optimizer.zero_grad()
            mu_v, var_v, value_v = net(states_v)

            loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

            adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
            log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
            loss_policy_v = -log_prob_v.mean()
            entropy_loss_v = ENTROPY_BETA * (-(torch.log(2 * math.pi * var_v) + 1) / 2).mean()

            loss_v = loss_policy_v + entropy_loss_v + loss_value_v
            loss_v.backward()
            optimizer.step()
        else:
            # Our optimization functions
            LM = MultiClassLM.LM()
    return best_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actiontype", required=True, help='Discrete or Continuous')
    parser.add_argument("--env", required=True, help="Env")
    parser.add_argument("--testenv", required=True, help="Test Env")
    parser.add_argument("--cuda", required=True, default=False, help="Cuda")
    parser.add_argument("--gamma",required=False,default=0.99)
    parser.add_argument("--lr", required=False, default=5e-5)
    parser.add_argument("--entropy",required=False,default=1e-4)
    args = parser.parse_args()

    action_type = args["actiontype"]
    env = args["env"]
    test_env = args["testenv"]
    cuda = args["cuda"]
    GAMMA = args["gamma"]
    LEARNING_RATE = args["lr"]
    ENTROPY_BETA = args["entropy"]
    device = torch.device("cuda" if cuda else "cpu")
    run_ac2(action_type,env,test_env,device,GAMMA,LEARNING_RATE,ENTROPY_BETA)



