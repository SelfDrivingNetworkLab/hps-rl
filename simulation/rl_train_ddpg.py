import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas
import time
import argparse
import numpy as np
from drllib.optimizations import MultiClassLM


from drllib import dynamic_models as models, utils, common

import torch
import torch.optim as optim
import torch.nn.functional as F

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

TEST_ITERS = 1000
RunName = "Test5"


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = utils.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def run_ddpq(actiontype, env, test_env, n_hidden, featureList, device, gamma, opt_func, batch, step, alr, clr,
             alp_reward, beta_reward, gamma_reward):
    # different lr values are assigned so general LR is  removed

    save_path = os.path.join("saves", "ddpg-" + RunName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_reward = []
    test_steps = []
    if actiontype == "Discrete":
        if env.action_space.n == 2:
            act_net = models.DDPGActor(env.observation_space.shape[0], env.action_space.n - 1, n_hidden,
                                       featureList).to(device)  # error here - problem when changing Gym
            crt_net = models.DDPGCritic(env.observation_space.shape[0], env.action_space.n - 1).to(
                device)  # solution: discrete type doesn't have shape, we need to use n
        else:
            act_net = models.DDPGActor(env.observation_space.shape[0], env.action_space.n, n_hidden, featureList).to(
                device)  # error here - problem when changing Gym
            crt_net = models.DDPGCritic(env.observation_space.shape[0], env.action_space.n).to(
                device)  # solution: discrete type doesn't have shape, we need to use n

    else:
        act_net = models.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(
            device)
        crt_net = models.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(
            device)

    tgt_act_net = utils.TargetNet(act_net)
    tgt_crt_net = utils.TargetNet(crt_net)

    # writer = SummaryWriter(comment="-ddpg_" + args.name)
    agent = models.AgentDDPG(act_net, device=device)
    exp_source = utils.ExperienceSourceFirstLast(env, agent, gamma=gamma, steps_count=1)
    buffer = utils.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)

    if opt_func is None or opt_func == "adam":
        act_opt = optim.Adam(act_net.parameters(), lr=alr)
        crt_opt = optim.Adam(crt_net.parameters(), lr=clr)

    frame_idx = 0
    best_reward = None
    
    #while True:
    for i in range(1000):
        frame_idx += 1
        buffer.populate(1)
        rewards_steps = exp_source.pop_rewards_steps()
        #print("REWARDS")
        #print(rewards_steps)
        if rewards_steps:
            rewards, steps = zip(*rewards_steps)

            if len(buffer) < REPLAY_INITIAL:
                continue

            batch = buffer.sample(batch)
            states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)
            if crt_opt:
                # train critic
                crt_opt.zero_grad()
                print(crt_net)
                print(states_v.shape)
                print(actions_v.shape)
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * gamma
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
            else:
                # Our optimization functions
                LM = MultiClassLM.LM()
            if act_opt:
                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
            else:
                # Our optimization functions
                LM = MultiClassLM.LM()

            tgt_act_net.alpha_sync(alpha=1 - alp_reward)
            tgt_crt_net.alpha_sync(alpha=1 - alp_reward)

            if frame_idx % TEST_ITERS == 0:
                ts = time.time()
                rewards, steps = test_net(act_net, test_env, device=device)
                print("Test done in %.2f sec, reward %.3f, steps %d" % (
                    time.time() - ts, rewards, steps))
                # writer.add_scalar("test_reward", rewards, frame_idx)
                # writer.add_scalar("test_steps", steps, frame_idx)
                test_reward.append(rewards)
                test_steps.append(steps)

                #print("ALL Rewards")
                #print(rewards)
                if best_reward is None or best_reward < rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(act_net.state_dict(), fname)
                        dftestrew = pandas.DataFrame(data={"test_reward": test_reward, "test_step": test_steps})
                        csv_name = os.path.join(save_path, "dftestrew.csv")
                        dftestrew.to_csv(csv_name, sep=',', index=False)

                    best_reward = rewards
                #print("BEST Rewards")
                #print(best_reward)
                
            #print("ALL Rewards")
            #print(rewards)  
            
    if best_reward == None: 
        return 0
    return best_reward
    #pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actiontype", required=True, help='Discrete or Continuous')
    parser.add_argument("--env", required=True, help="Env")
    parser.add_argument("--testenv", required=True, help="Test Env")
    parser.add_argument("--cuda", required=True, default=False, help="Cuda")
    parser.add_argument("--nhidden", required=False, default=1)
    parser.add_argument("--features", required=False, default=[300, 400])
    args = parser.parse_args()

    action_type = args["actiontype"]
    env = args["env"]
    test_env = args["testenv"]
    cuda = args["cuda"]
    nhidden = args["nhidden"]
    features = args["features"]
    device = torch.device("cuda" if cuda else "cpu")

