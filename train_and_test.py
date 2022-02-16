from Steed.gymlabs.envs.cartpole_env import CartPoleEnv
import Steed.simulation.rl_train_ddpg as ddpg
import Steed.simulation.rl_play_ddpg as playddpg
import Steed.simulation.rl_train_a2c as a2c
import Steed.simulation.rl_play_a2c as playa2c
import Steed.simulation.rl_play_aktr as playaktr
import Steed.simulation.rl_train_aktr as aktr
import Steed.simulation.rl_play_trpo as playtrpo
import Steed.simulation.rl_train_trpo as trpo
import torch
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

CUDA=False
device = torch.device("cuda" if CUDA else "cpu")
##### Environment calling
env = CartPoleEnv()
test_env = CartPoleEnv()
##########################HPO#########################################


ddpg.run_ddpq("Discrete",env,test_env,1,[400,300],device) #test for dynamic model
playddpg.run_ddpg("Discrete",env,device)
# aktr.run_aktr("Discrete",env,test_env,device)
# playaktr.run_aktr("Discrete",env,device)
# trpo.run_trpo("Discrete",env,test_env,device)
# playtrpo.run_trpo("Discrete",env,test_env,device)