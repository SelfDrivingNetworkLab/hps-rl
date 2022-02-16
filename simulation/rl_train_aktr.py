#import sys

#sys.path.append("//Users/mkiran/SWProjects/calibers/DRL_FLEXLAB/")

import argparse
import math
import os
import time

import numpy as np
import pandas
import torch
import torch.nn.functional as F
import torch.optim as optim

from Steed.drllib import model4trpo, utils, kfac, common
from Steed.optimizationfns import MultiClassLM

#QUESTION: Is model4trpo the same with trpomodel?

# testting commit


ENVS_COUNT = 16

TEST_ITERS = 5

RunName = "Test70"

def test_net(net, env, exp_idx, count=1, device="cpu"):
    # print(exp_idx)
    # shift = exp_idx + 50000
    
    rewards = 0.0
    e_costs = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        # t_idx = 0
        while True:
            # t_idx += 1
            print("in true test")
            obs_v = utils.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            #print("action")
            #print(action)
            obs, reward, done, divers = env.step(action)
            #print("done:")
            print(done)
            e_cost = divers[0]

            rewards += reward
            e_costs += e_cost
            steps += 1
            if done:
                break
    print("in test_net")
    print(rewards)
    return rewards / count, steps / count, e_costs / count


def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


def run_aktr(actiontype,env,test_env,device,GAMMA,OPT_FUNC,BATCH,ENTROPY_BETA,REWARD_STEPS, STEP, ALR, CLR, ALP_REWARD, BETA_REWARD, GAMMA_REWARD):

    advantage = []
    values = []
    loss_value = []
    batch_rewards = []
    loss_entropy = []
    loss_total = []
    loss_policy = []
    test_reward = []
    test_steps = []

    save_path = os.path.join("saves", "acktr-" + RunName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sim_days = 365
    eprice_ahead = 3
    alpha_r = 100.0
    beta_r = 4.5
    gamma_r = 1.5
    delta_r = 0.0
    pv_panels = 10.0
    light_ctrl = False

    if actiontype == "Discrete":
        if env.action_space.n == 2:
             net_act = model4trpo.ModelActor(env.observation_space.shape[0], env.action_space.n - 1).to(device)
        else:
            net_act = model4trpo.ModelActor(env.observation_space.shape[0], env.action_space.n).to(device)
    else:
        net_act = model4trpo.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    net_crt = model4trpo.ModelCritic(env.observation_space.shape[0]).to(device)


  #  net_act = tpro_model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
  #  net_crt = tpro_model.ModelCritic(env.observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    start_time = time.time()

    agent = model4trpo.AgentA2C(net_act, device=device)
    exp_source = utils.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    if OPT_FUNC is None or OPT_FUNC == "adam":
        opt_act = kfac.KFACOptimizer(net_act, lr=ALR)
        opt_crt = optim.Adam(net_crt.parameters(), lr=CLR)

    batch = []
    best_reward = None
    for step_idx, exp in enumerate(exp_source):
        print("******************************step_idx")
        print(step_idx)
        rewards_steps = exp_source.pop_rewards_steps()
        print("reward_steps")
        print(rewards_steps)

        if rewards_steps:
            print("new rewards")
            print(rewards_steps)
            rewards, steps = zip(*rewards_steps)

        print("env steps")
        print(env.n_steps)

        if step_idx % ((env.n_steps - 1) * TEST_ITERS) == 0:
            print("in step loop")

            ts = time.time()
            rewards, steps, e_costs = test_net(net_act, test_env, step_idx, device=device)
            print("Test done in %.2f sec, reward %.3f, steps %d" % (
                time.time() - ts, rewards, steps))
            test_reward.append(rewards)
            test_steps.append(steps)

            if best_reward is None or best_reward < rewards:
                print("*********************In saving loop")
                print(best_reward)

                best_reward = rewards
                print("new best rewards")
                print(best_reward)

                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                    fname = os.path.join(save_path, name)

                    torch.save(net_act.state_dict(), fname)
                    dfadvvalueloss = pandas.DataFrame(
                        data={"advantage": advantage, "values": values, "loss_values": loss_value})
                    dftestrew = pandas.DataFrame(data={"test_reward": test_reward, "test_step": test_steps})
                    dfadvvalueloss.to_csv("performancedata/dfadvvalueloss.csv", sep=',', index=False)
                    dftestrew.to_csv("performancedata/dftestrew.csv", sep=',', index=False)

                    dfnewaktr = pandas.DataFrame(
                        data={"batch_rewards": batch_rewards, "loss_entropy": loss_entropy,
                              "loss_total": loss_total, "loss_policy": loss_policy})
                    dfnewaktr.to_csv("performancedata/dfnewaktr.csv", sep=',', index=False)

        batch.append(exp)
        if len(batch) < BATCH:
            print(len(batch))

            continue

        states_v, actions_v, vals_ref_v = \
            common.unpack_batch_a2c(batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
        del batch[:]

        if opt_crt:
            opt_crt.zero_grad()
            value_v = net_crt(states_v)
            loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
            loss_value_v.backward()
            opt_crt.step()
        else:
            # Our optimization functions
            LM = MultiClassLM.LM()

        mu_v = net_act(states_v)
        log_prob_v = calc_logprob(mu_v, net_act.logstd, actions_v)

        if opt_act:
            if opt_act.steps % opt_act.Ts == 0:
                opt_act.zero_grad()
                pg_fisher_loss = -log_prob_v.mean()
                opt_act.acc_stats = True
                pg_fisher_loss.backward(retain_graph=True)
                opt_act.acc_stats = False

            opt_act.zero_grad()
            adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
            loss_policy_v = -(adv_v * log_prob_v).mean()
            entropy_loss_v = ENTROPY_BETA * (-(torch.log(2 * math.pi * torch.exp(net_act.logstd)) + 1) / 2).mean()
            loss_v = loss_policy_v + entropy_loss_v
            loss_v.backward()
            opt_act.step()
        else:
            # Our optimization functions
            LM = MultiClassLM.LM()

        print("###########################################End of main")
        print("step : %d" % step_idx)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actiontype", required=True, help='Discrete or Continuous')
    parser.add_argument("--env", required=True, help="Env")
    parser.add_argument("--testenv", required=True, help="Test Env")
    parser.add_argument("--cuda", required=True, default=False, help="Cuda")
    args = parser.parse_args()

    action_type = args["actiontype"]
    env = args["env"]
    test_env = args["testenv"]
    cuda = args["cuda"]
    device = torch.device("cuda" if cuda else "cpu")
    run_aktr(action_type, env, test_env, device)

