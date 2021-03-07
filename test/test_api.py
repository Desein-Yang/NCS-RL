#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_api.py
@Time    :   2020/04/19 08:27:37
@Version :   1.0
@Describtion:  Test API of policy and optimizer
'''

# here put the import lib
import gym
import numpy as np
from logger import Logger
from policy import Policy
from testfunc import test_func_bound,TestEnv
from policy_func import FuncPolicy

args = {
    "optimizer": "NCSOptimizer",
    "settings": {
    "learning_rate": 1,
    "sigma": 0.01,
    "c_sigma_factor": 1,
    "mu": 50
    },
    "network": "Nature",
    "nonlin_name": "relu",

    "group_cpus":3,
    "eva_times":1,
    "game":'Freeway',
    "run_name":'debug',
    "epoch": 5,
    "r":0.2,
    "timesteps_limit": 1e8,
    "D":1700000, # model size need to be set in run
    "d":10000,
    "m":10,
    "mu":3
}

def test_log():
    logger = Logger("./test_log/")
    logger.log("Hello world")
    logger.write_general_stat("stat string")
    logger.write_optimizer_stat("stat string")

def test_policy(env_type):
    logger = Logger("./test_log/")
    if env_type == "atari":
        game = "Alien"
        env = gym.make("%sNoFrameskip-v4" % game)
        policy = Policy(env, network=args['network'], nonlin_name=args['nonlin_name'])
        vb = policy.get_vb()
        policy.set_vb(vb)
        logger.save_vb(vb)
    elif env_type == "function":
        env = TestEnv(10000,1)
        policy = FuncPolicy(10000,1,0,1)
    params = policy.get_parameters()
    print(params.shape)
    rs = np.random.RandomState(100000)
    A = rs.standard_normal((5,params.shape[0]))
    # A narray .shape = [5,10000] size = 50000
    # params narray .shape = (10000,) size = 10000
    # effparam narray .shape = (5,) size = 5
    print(A)
    eff_params = np.dot(A,params)
    print(eff_params)
    d = eff_params.shape[0]
    policy.set_parameters(params)
    params = policy.get_parameters()

if __name__ == "__main__":
    # test_log()
    test_policy("function")
