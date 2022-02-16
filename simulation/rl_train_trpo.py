#!/usr/bin/env python3
import os
import math
import time
import gym
import argparse
import pandas


from Steed.drllib import model4trpo, trpo, utils

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F



GAE_LAMBDA = 0.95
TRPO_DAMPING = 0.1  #dont play with this

TEST_ITERS = 5
RunName = "Test60"


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
            obs_v = utils.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)

            obs, reward, done, divers = env.step(action)

            e_cost = divers[0]

            rewards += reward
            e_costs += e_cost
            steps += 1
            if done:
                break
    return rewards / count, steps / count, e_costs / count



def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


def calc_adv_ref(trajectory, net_crt, states_v, GAMMA, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: list of Experience objects
    :param net_crt: critic network
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v

def run_trpo(actiontype,env,test_env,device,TRAJECTORY_SIZE,GAMMA,CLR,MAX_KL,STEP):
    # save all values in dataframe:

    advantage = []
    values = []
    loss_value = []
    episode_steps = []
    test_reward = []
    test_steps = []

    save_path = os.path.join("saves", "trpo-" + RunName)
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

    print(net_act)
    print(net_crt)
    start_time = time.time()


    agent = model4trpo.AgentA2C(net_act, device=device)
    exp_source = utils.ExperienceSource(env, agent, steps_count=1)

    opt_crt = optim.Adam(net_crt.parameters(), lr=CLR)

    trajectory = []
    best_reward = None
    for step_idx, exp in enumerate(exp_source):
            # print("******************************step_idx")
            # print(step_idx)
            rewards_steps = exp_source.pop_rewards_steps()
            # print("reward_steps")
            # print(rewards_steps)
            if rewards_steps:
                # print("new rewards")
                # print(rewards_steps)
                rewards, steps = zip(*rewards_steps)
                episode_steps.append(np.mean(steps))
            value = (step_idx % ((env.n_steps - 1) * TEST_ITERS))

            if (step_idx % ((env.n_steps - 1) * TEST_ITERS)) == 0:
                # print("in step loop")
                ts = time.time()
                # rewards, steps = test_net(net_act, test_env, device=device)
                rewards, steps, e_costs = test_net(net_act, test_env, step_idx, device=device)

                print("Test done in %.2f sec, reward %.3f, steps %d" % (
                    time.time() - ts, rewards, steps))
                test_reward.append(rewards)
                test_steps.append(steps)
                if best_reward is None or best_reward < rewards:
                    print("*********************In saving loop")
                    print(best_reward)
                    # moving this hear tosave the first nn
                    best_reward = rewards
                    print("new best rewards")
                    print(best_reward)

                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(net_act.state_dict(), fname)
                        # save dataframe to files

                        dfadvvalueloss = pandas.DataFrame(
                            data={"advantage": advantage, "values": values, "loss_values": loss_value})
                        dftestrew = pandas.DataFrame(data={"test_reward": test_reward, "test_step": test_steps})
                        dfadvvalueloss.to_csv("performancedata/dfadvvalueloss.csv", sep=',', index=False)
                        dftestrew.to_csv("performancedata/dftestrew.csv", sep=',', index=False)

            trajectory.append(exp)
            # print("*******************trajectory len")
            # print(len(trajectory))

            if len(trajectory) < TRAJECTORY_SIZE:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            # print("trajactions")
            # print(traj_actions)
            traj_states_v = torch.FloatTensor(traj_states).to(device)
            # print("1")
            traj_actions_v = torch.FloatTensor(traj_actions).to(device)
            # print("2")
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, GAMMA, device=device)
            # print("3")
            mu_v = net_act(traj_states_v)
            # print("4")
            old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)
            # print("5")
            # normalize advantages
            traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)
            # print("6")
            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            # print("7")
            old_logprob_v = old_logprob_v[:-1].detach()
            traj_states_v = traj_states_v[:-1]
            traj_actions_v = traj_actions_v[:-1]
            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0
            # print("8")

            # critic step
            opt_crt.zero_grad()
            value_v = net_crt(traj_states_v)
            loss_value_v = F.mse_loss(value_v.squeeze(-1), traj_ref_v)
            loss_value_v.backward()
            opt_crt.step()
            def get_loss():
                mu_v = net_act(traj_states_v)
                logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)
                action_loss_v = -traj_adv_v.unsqueeze(dim=-1) * torch.exp(logprob_v - old_logprob_v)
                return action_loss_v.mean()

            def get_kl():
                mu_v = net_act(traj_states_v)
                logstd_v = net_act.logstd
                mu0_v = mu_v.detach()
                logstd0_v = logstd_v.detach()
                std_v = torch.exp(logstd_v)
                std0_v = std_v.detach()
                kl = logstd_v - logstd0_v + (std0_v ** 2 + (mu0_v - mu_v) ** 2) / (2.0 * std_v ** 2) - 0.5

                return kl.sum(1, keepdim=True)


            trpo.trpo_step(net_act, get_loss, get_kl, MAX_KL, TRPO_DAMPING, device=device)

            del trajectory[:]  # or use trajectory=[]
            advantage.append(traj_adv_v.mean().item())
            values.append(traj_ref_v.mean().item())
            loss_value.append(loss_value_v.item())

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
    run_trpo(action_type,env,test_env,device)