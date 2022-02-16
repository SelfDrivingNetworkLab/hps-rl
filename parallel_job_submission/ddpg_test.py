import sys
from gymlabs.envs.cartpole_env import CartPoleEnv
import simulation.rl_train_ddpg as ddpg
import simulation.rl_play_ddpg as playddpg
import torch
import GA4RL as GA4RL
import os
#SLURM FILE#

action_type = "DISCRETE" #DISCRETE


CUDA=False
device = torch.device("cuda" if CUDA else "cpu")
##### Environment calling
env = CartPoleEnv()
test_env = CartPoleEnv()
##########################HPO#########################################
def run_job():
    GA = GA4RL.GA4RL(model_name="DDPG", action_type=action_type, env=env, test_env=test_env, device="cpu", opt_func="LM",
           nof_generations=3, pop_size=50, nof_elites=1, crossover_rate=0.7, mutation_prob=0.05)
    GA.Run(numProcesses=2, numThreads=4)
    best, flog = GA.Run()
    #  parameter_list = ["gamma", "learning_rate", "hidden_layers", "nodes_per_layer", "batch_size", "step_size",
#                              "actor_learning_rate","critic_learning_rate","alpha_reward","beta_reward","gamma_reward",
#                           "test_iteration"]
    ddpg.run_ddpq(action_type, env, test_env, best[2], best[3], device, best[0], "LM", best[4], best[5], best[6], best[7],
            best[8], best[9], best[10])

    playddpg.run_ddpg("Discrete",env,device)


if os.name == 'nt':
  if __name__ == '__main__':
        run_job()

