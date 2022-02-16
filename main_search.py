#This is the main file you run to search for optimal hyperparameters


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import random
import pandas
import time
import argparse
import numpy as np
import copy
#from Steed.optimizationfns import MultiClassLM

#from Steed.drllib import dynamic_models as models, utils, common
from Steed.gymlabs.envs.cartpole_env import CartPoleEnv


import torch
import torch.optim as optim
import torch.nn.functional as F

#set a constant buffer size
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
#set a constant test iter size, this determine stopping conditions of individual unit
TEST_ITERS = 1000

#set your algorithm conditions to be searched 
#DQN,DDPG,TRPO,AKTR, A2C
ALGO="DDPG"

SEED=50
EACHTRIAL = 5 #each combination tried, returns average reward
#GAMMA=0.01 # discount factor user sets between 0 immediate rewards (small for small spaces) and 1 delayed rewards(high for large spaces)
ALPHA=0.9  # LR values is set to decay by default as RL trains high means learning to quickly
ENV_ID=CartPoleEnv()
----userdefine alphadecay

#set search parameters
ACTFN=["TANH", "RELU"]
OPTFN=["ADAM","CG","LBFGS", "LM"]

#MK: ASK about this :Melis set ALP, BLP, GLP, we need this? -user defined
GAMMA =[0.01, 0.1,0.5,0.99]
#neural network architecture (features+hidden+hidden+outputs)
NODESLAYER=[10, 50, 100, 128]
ALPHA=[0.1, 0.01, 0.001] #LR actor and critic
ACTOR_ALPHA=[0.1, 0.01, 0.001]
CRITIC_ALPHA=[0.1, 0.01, 0.001]

#A2C AKTR
ENTROPY_BETA=[0.001,0.01,0.1] #used a2c, aktr

#TRPO/AKTR
TRAJECTORY_SIZE=[10, 20, 50, 100, 1000]
MAX_KL=[0.001, 0.01, 0.1]
GAE_Lambda=[0.1,0.0001]


GENERATIONS=20
POPSIZE=20
CROSSOVER_RATE=0.5
MUTATION_RATE =0.1
CHROMOSOME_LENGTH=10

class Ddpg_Parent():
    def __init__(self, gamma,actfn,optfn,nodes,alr,clr,frequency,score):
        self.gamma=gamma
        self.actfn=actfn
        self.optfn=optfn
        self.nodes=nodes
        self.alr=alr 
        self.clr=clr
        self.frequency=frequency
        self.score=score

    def ret_gamma(self):
        return self.gamma

    def ret_score(self):
        return self.score



if __name__=="__main__":

    print("Initializing search for hyperparameters")
    random.seed(SEED)
    population_queue=[]
    #initial the population
    for gen in range(POPSIZE):
        random_gamma=random.randint(1,len(GAMMA))-1
        print(random_gamma)
        random_actfn=random.randint(1,len(ACTFN))-1
        random_optfn=random.randint(1,len(OPTFN))-1
        random_nodenum=random.randint(1,len(NODESLAYER))-1
        random_actorlr=random.randint(1,len(ACTOR_ALPHA))-1
        random_criticlr=random.randint(1, len(CRITIC_ALPHA))-1
        person=Ddpg_Parent(GAMMA[random_gamma], ACTFN[random_actfn],OPTFN[random_optfn],
        NODESLAYER[random_nodenum],ACTOR_ALPHA[random_actorlr],CRITIC_ALPHA[random_criticlr],
        0.0,0.0)
        population_queue.append(person)
    #check
    for a in range(len(population_queue)):
        #n=Ddpg_Parent()
        #n=a
        print(population_queue[a].ret_gamma())

    sum_score=0.0
    sum_prob=0.0
    #Perform parent selection
    for a in range(len(population_queue)):
        sum_score+=population_queue[a].ret_score()

    for a in range(len(population_queue)):
        sum_prob+=population_queue[a].ret_score()/sum_score

    parent1=Ddpg_Parent(0.0, 0.0,0.0,0.0,0.0,0.0,
        0.0,0.0)

    start_ptr=random_uniform(0,1)

    for a in range(len(population_queue)):



