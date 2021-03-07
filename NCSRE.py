#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   NCSRE-v3.py
@Time    :   2020/07/27 09:11:10
@Author  :   Qi Yang
@Version :   2.0
@Describtion:  NCS为主体RE为方法的版本v2 sigma 衰减 可运行
"""

# here put the import lib
import numpy as np
import gym
import time
import os
import pickle
import warnings
import cProfile
import json
import click
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from mpi4py import MPI
from src.logger import Logger
from src.policy import Policy,FuncPolicy
from src.env_wrappers import wrap_dqn
from src.testfunc import test_func_bound
<<<<<<< HEAD
#from memory_profiler import profile
#from memory_profiler import memory_usage
=======
from memory_profiler import profile
from memory_profiler import memory_usage
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

"""This class is a subprocess in main loop.
Group rank is rank in NCS.
Group id is the id of all NCS populations.
Rank 0 is responsible to log, test, gather and bcast params.
<<<<<<< HEAD
Rank 1-6 
=======
Rank 1-6
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
Rank 7-12...."""


class NCSREAlgo(object):
    def __init__(self, args):
        # Hyper
        self.args = args
        self.D = 0  # model size will be reset
        self.d = args["d"]
        self.lam = args["lam"]
        self.mu = args["mu"]
        self.episode = args["m"]  # m : number of RE phase
<<<<<<< HEAD
        self.len_phase = args["len_phase"] 
=======
        self.len_phase = args["len_phase"]
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        self.steps_max = args["steps_max"]
        # self.group_cpus = args["group_cpus"]
        self.group_cpus = self.lam
        self.L , self.H = (-0.1,0.1)
        self.L_, self.H_ = (-5,5)
        self.eps = 1e8
        self.sigma_reup = args["sigma_reup"]

        # ncs
        self.llambda = args["lambda0"]
        self.epoch = args["epoch"]
        self.k = args["k"]
        self.r = args["r"]
        self.algoname = 'NCSRE'
        self.steps_passed = 0
        self.last_retest_steps = 0

        # MPI
        self.comm = MPI.COMM_WORLD
        self.id = self.comm.Get_rank()
        self.cpus = self.comm.Get_size()
        #assert self.cpus == (1 + self.group_cpus * self.mu),"Cpu assignment Error"
        self.train_set, self.test_set = self.getSeedPool(self.args['n_train'],self.args['n_test'],self.randomSeed


        if self.id != 0:
            self.rank = (self.id - 1) % self.group_cpus
            self.group_id = int(np.floor((self.id - 1) / self.group_cpus))
        else:
            self.rank = 0
            self.group_id = self.mu

        # e.g. id = 1 group = 0 id = 10 group = 9/5 = 1
        # random matrix
        if args["seed"] == 0:
            self.random_seed = np.random.randint(100000) + self.id
        else:
            self.random_seed = args["seed"] + self.id
        # np.random.seed(self.random_seed)
        # use seed to convey and save matrix
        self.seeds = self.getSeed()
        self.slice_len = args["slice_len"]
<<<<<<< HEAD
        
=======
        if self.args["seed_pool"] == True:
            self.train_set, self.test_set = self.getSeedPool(self.args['n_train'],self.args['n_test'],self.random_seed)
        else:
            self.train_set = np.arange(1e6,dtype=np.int)
            self.test_set = self.train_set

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        # logger
        self.logger = Logger(self.logPath())

        # env
        if self.args["env_type"] == "atari":
            env = gym.make("%sNoFrameskip-v4" % self.args["game"])
            print("make env")
            human = 1 if self.args["human_start"] is True else 0
            env = wrap_dqn(env,human_start=human)

            self.policy = Policy(
                env, network=args["network"],
                nonlin_name=args["nonlin_name"]
            )
            vb = self.policy.get_vb()
            self.comm.Bcast([vb, MPI.FLOAT], root=0)
            self.policy.set_vb(vb)
            self.logger.save_vb(vb)
<<<<<<< HEAD
            self.steps_max = args["steps_max"] / 4
            self.sigma0 = args["sigma0"]
        elif self.args["env_type"] == "function":
            self.policy = FuncPolicy(
                self.args["func_dim"], self.args["func_id"], 
=======
            self.sigma0 = args["sigma0"]
        elif self.args["env_type"] == "function":
            self.policy = FuncPolicy(
                self.args["func_dim"], self.args["func_id"],
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
                self.rank, self.random_seed
            )
            self.steps_max = 5000 * self.args["func_dim"]
            bound = test_func_bound[self.args["func_id"]]
<<<<<<< HEAD
            self.sigma0 = (bound[1] - bound[0]) / self.group_cpus
=======
            #self.sigma0 = (bound[1] - bound[0]) / self.group_cpus

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

        self.sigma = np.ones((self.d,), dtype = np.double) * self.sigma0
        self.sigma_all = np.empty((self.cpus, self.d), dtype = np.double)
        self.comm.Allgather([self.sigma, MPI.DOUBLE],
                            [self.sigma_all, MPI.DOUBLE])

        # withdraw factor
        self.alpha = np.ones(1) * args['alpha0']
        self.alpha_all = np.empty((self.cpus, 1))
        self.comm.Allgather([self.alpha, MPI.DOUBLE],
                            [self.alpha_all,MPI.DOUBLE])

        # param is for rank ! = 0
        # param_all is for all
        self.eff_params = np.random.uniform(self.L,self.H,(self.d,))
        self.eff_params_all = np.empty((self.cpus,self.d))
        self.comm.Allgather(
            [self.eff_params, MPI.DOUBLE],
            [self.eff_params_all, MPI.DOUBLE]
        )
        self.eff_params_new_all = self.eff_params_all.copy()
        self.param = self.policy.get_parameters()
        self.param = np.clip(self.param,self.L_,self.H_)
        self.D = len(self.param)
        self.params_all = np.empty((self.cpus,self.D))

        self.steps_passed = 0
        self.last_retest_steps = 0
        self.iter = 1
        self.phase = 1

        # NCS line 12-13 best params x,y and score of every ncs 12-13
        self.Bestscore = np.zeros(1)
        self.Bestscore_all = np.zeros((self.cpus, 1))

        self.Besteffparams = self.eff_params.copy()
        self.Besteffparams_all = self.eff_params_all.copy()
        self.Bestparams = self.param.copy()
        self.Bestparams_all = np.empty((self.cpus,self.D))
        # NCSRE line 10 , best local score and best params x
        self.LocalBestscores = np.zeros((self.lam, 1))
        self.LocalBestparams = np.random.uniform(self.L_,self.H_,(self.lam, self.D,))
        self.LocalBesteffparams = np.random.uniform(self.L,self.H,(self.lam, self.d,))

        # NCSRE line 12 , best params x in history and all local subprocess
        self.GlobalBestscore = np.zeros(1)
        self.GlobalBestparam = np.zeros((self.D,))
        self.GlobalBesteffparam = np.zeros((self.d,))

        self.logBasic()

        self.log_retest = {
            "steps": [],
            "performance": []
        }
        self.log_train = {
            "steps": [],
            "performance": []
        }
        #self.log_detail = {
        #    "id" : self.id,
        #    "corr_new": [],
        #    "corr_old": [],
        #    "eff_params": [], # [(max,min,mean,sigma),]
        #    "params": [],
        #    "sigma": [],
        #    "succ" : [],
        #    "lambda":[]
        #}

        self.firstEvaluation()
        self.child_reward = np.zeros((self.cpus,),dtype='d')


    def firstEvaluation(self):
        """evaluation for first initialization
        """
        msg = self.rollout(self.param)
        reward_child = msg[0]
        # results = np.empty((self.cpus, 2))
        self.child_reward = self.syncOneValue(reward_child)
        self.father_reward = self.child_reward.copy()
        self.updateBest()
        if self.id == 0:
            self.seeds = self.getSeed()
        else:
            self.seeds = np.empty((self.cpus,),dtype = 'i')
        self.comm.Bcast([self.seeds, MPI.INT],root = 0)

        self.updateLocalbest()
        self.updateGlobalbest()
        # self.retest()

    def getSeedPool(self,n_train,n_test,seed=None,zero_shot=True,range=1e5):
<<<<<<< HEAD
        """Create train and test random seed pool.  
=======
        """Create train and test random seed pool.
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        If use zero shot performance, test seed will non-repeatitive.  """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.default_rng()
        Allseed = np.arange(range,dtype=np.int)
        trainset = rng.choice(Allseed,size=n_train,replace=False)  # means sample without replacement
        tmp = np.setdiff1d(Allseed,trainset,assume_unique=True) # get disjoint set of trainset
        if zero_shot is False:
            testset = rng.choice(tmp,size=n_test,replace=False)
        else:
            testset = rng.choice(Allseed,size=n_test,replace=False)
        return trainset, testset

<<<<<<< HEAD
    #@profile    
=======
    #@profile
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
    def run(self):
        """Main loop of NCSRE.  """
        self.start_time = time.time()

        # self.logger.log_for_debug("Start Main Loop 205")
        while self.steps_passed <= self.steps_max:

            phase_start_time = time.time()
            # Init sigma every phase
            # Maybe change
            if self.sigma_reup == 1:
                self.sigma = np.ones(self.d, dtype = np.double) * self.sigma0
                self.comm.Allgather([self.sigma, MPI.DOUBLE],
                                [self.sigma_all, MPI.DOUBLE])
            self.updatecount = 0
            # self.logger.log_for_debug("Sigma set 213")

            if self.id == 0:
                self.seeds = self.getSeed()
            else:
                self.seeds = np.empty((self.cpus,),dtype = 'i')
            self.comm.Bcast([self.seeds, MPI.INT],root = 0)
            assert self.seeds is not None, 'Seed set Error'

            # Local best params is best among different embedding space
            # rank 0 - lam get corresponding best
            self.param = self.LocalBestparams[self.rank].copy()
            #self.eff_params = self.LocalBesteffparams[self.rank].copy()
            self.eff_params = np.random.uniform(self.L,self.H,(self.d,))
            self.logger.log_for_debug(" Run : Ini params and effparams 224")
            self.logger.log_for_debug(' 217 params '+str(self.calDist(self.param))+' '+str(self.id))
            self.logger.log_for_debug(' 218 eff params'+str(self.calDist(self.eff_params))+' '+str(self.id))


            phase_cost_steps = self.optimize()  # best params has been synced
            self.steps_passed += phase_cost_steps
            # Notes: now we use only 1 best will be pick
            # So update global best is empty and pass
            self.updateLocalbest()
            self.updateGlobalbest()
            self.retest()

            self.logRE(phase_start_time,phase_cost_steps)
            self.phase += 1
        #self.saveDetail()
        self.saveRetestLog()
<<<<<<< HEAD
            
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
    #@profile
    def optimize(self):
        """ NCS Main loop.  """
        phase_cost_steps = 0
        while self.iter <= self.phase * self.len_phase:
            iter_start_time = time.time()
            if self.iter % self.epoch == 0:
                self.updatecount = 0

            # Notes: group id in rank0 is None, so reward and step in rank
            # will be empty and may lead to bug in MPI gather
            # MCS 8-9 Generate new child and eval
            cost_steps = self.genEvalChild()

            self.updateBest()
            self.updateFather()

            phase_cost_steps += cost_steps

            if self.iter % self.epoch == 0:
                self.updateSigma()
                self.comm.Allgather(
                    [self.sigma, MPI.DOUBLE],
                    [self.sigma_all, MPI.DOUBLE]
                )

            #self.log_detail['sigma'].append(self.sigma)
            #self.log_detail['corr_new'].append(self.corr_new)
            #self.log_detail['succ'].append(self.updatecount)

            self.logNCS(iter_start_time, cost_steps)
            self.iter += 1

        return phase_cost_steps

    def genChild(self):
        """ generate a child on y"""
        pass

    def genEvalChild(self):
        """ generate a child on y
        Return:
            cost_steps
        Modified:
            corr_new, corr_old, reward_child
        """
        cost_steps = 0
        if self.id != 0:
            # set lambda ncs 6
            np.random.seed(self.random_seed+self.iter)
            self.llambda = (
                np.random.randn()
                * (0.1 - 0.1 * self.steps_passed / self.steps_max)
                + 1.0)

            # y' = y + noise, ncs 8
            np.random.seed( self.random_seed + self.iter + 1 )
            #self.logger.log_for_debug('297 eff params before noise '+str(self.calDist(self.eff_params)))
            eff_params_new = (
                self.eff_params.copy() +
                np.random.randn(self.d) * self.sigma
            )
            eff_params_new = np.clip(eff_params_new,self.L,self.H)
            #self.logger.log_for_debug('393 eff params after noises'+str(self.calDist(eff_params_new)))
            assert eff_params_new.shape == (self.d,), 'Eff_param_new shape Error'

            params_new = self.from_y_to_x(eff_params_new)
            params_new = np.clip(params_new,self.L_,self.H_)

            if self.args["env_type"] == "function":
                bound = test_func_bound[self.args["func_id"]]
                params_new[params_new < bound[0]] = bound[0]
                params_new[params_new > bound[1]] = bound[1]

            # calculate fitness of child, ncs 9
            assert len(params_new) == self.D, 'Params_new set error'
            msg_new = self.rollout(params_new,test=False)
            reward_child = msg_new[0]
            steps_cost_child = msg_new[1]
        else:
            reward_child = np.float(0.0)
            steps_cost_child = 0
            params_new = self.param.copy()
            eff_params_new = self.eff_params.copy()

        # synchronize params to get distribution ncs 8-9
        self.sigma_all = self.syncOneVector(self.sigma)
        self.eff_params_all = self.syncOneVector(self.eff_params)
        self.params_all = self.syncOneVector(params_new)
        self.eff_params_new_all = self.syncOneVector(eff_params_new)
        self.child_reward = self.syncOneValue(reward_child)
        # Notes: some ind didn't finish rollout but some has done
        # it's sync

        # calculate correlation of child and father,ncs 9
        if self.id != 0:
            corr_old = self.calCorr(
                self.eff_params_all, self.eff_params,
                self.sigma_all, self.sigma
            )
            corr_new = self.calCorr(
                self.eff_params_new_all, eff_params_new,
                self.sigma_all, self.sigma
            )
            self.corr_new = corr_new / (corr_new + corr_old  + self.eps)
            self.corr_old = corr_old / (corr_new + corr_old  + self.eps)
        else:
            self.corr_new = 0
            self.corr_old = 0

        tmp_step = self.syncOneValue(steps_cost_child)
        cost_steps += int(np.sum(tmp_step))
        return cost_steps

    # finished
    def updateBest(self):
        """Update bestfound in single ncs process ncs 12-13.
        update local bestfound of each x"""
        # I guess it call same bestscore so it can't sync
        # and cause error in mpi
        # score has been sync to self.child reward
        if self.id != 0:
            score = self.child_reward[self.id]
            if score > self.Bestscore[0]:
                self.Bestscore[0] = score
                #self.logger.log_for_debug("update Best"+str(self.Bestscore)+str{self.id})
                self.Besteffparams = self.eff_params_new_all[self.id].copy()
                self.Bestparams = self.params_all[self.id].copy()
        else:
            self.Bestscore[0] = 0.0
            self.Besteffparams = self.eff_params.copy()
            self.Bestparams = self.param

    def updateLocalbest(self):
        """ ncsre 9-11 sycn NCS's BESTFOUND.
        has been tested.

        v1: return one bestfound in all groups.
        v2: return lambda bestfound in each random matrix.
        now I use v2 and it is global.

        """
        self.comm.Allgather([self.Bestscore        ,MPI.DOUBLE],
                            [self.Bestscore_all    ,MPI.DOUBLE])
        self.comm.Allgather([self.Besteffparams    ,MPI.DOUBLE],
                            [self.Besteffparams_all,MPI.DOUBLE])
        self.comm.Allgather([self.Bestparams       ,MPI.DOUBLE],
                            [self.Bestparams_all   ,MPI.DOUBLE])
        if self.id == 0:
            for i,score in enumerate(self.Bestscore_all[1:]):
                # Notes: cpu 0 is not included
                # has been test
                rank = i % self.group_cpus
                if score > self.LocalBestscores[rank]:
                    self.LocalBestscores[rank] = score
                    self.LocalBesteffparams[rank] = self.Besteffparams_all[i+1].copy()
                    self.LocalBestparams[rank] = self.Bestparams_all[i+1].copy()
                    self.logger.log_for_debug("Update Local Best 417 LocalBestscore %d %s" % (self.id,self.LocalBestscores))
                    self.logger.log_for_debug('Update local best LocalBesteffparams '+str(self.calDist(self.LocalBesteffparams[rank])))
                    self.logger.log_for_debug('Update local best LocalBestparams '+str(self.calDist(self.LocalBestparams[rank])))
            self.log_train["steps"].append(self.steps_passed)
            self.log_train["performance"].append(self.LocalBestscores)
            print("Train Steps %s".ljust(25) % str(self.steps_passed))
            print("LocalBestscore %s".ljust(25) % str(self.LocalBestscores.flatten()))

        self.comm.Bcast([self.LocalBestscores   ,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.LocalBesteffparams,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.LocalBestparams   ,MPI.DOUBLE], root = 0)

    def updateGlobalbest(self):
        """同步全局NCS的BESTFOUND"""
        if self.id == 0:
            self.GlobalBestscore = np.max(self.LocalBestscores.flatten())
            idx = np.argmax(self.LocalBestscores.flatten())
            self.GlobalBestparam = self.LocalBestparams[idx].copy()
            self.logger.log_for_debug("best id"+str(idx))
            self.logger.log_for_debug("Update Global Best 416 %d %s" %(self.id,self.GlobalBestscore))
            self.logger.log_for_debug("Update Global Best 416" + str(self.calDist(self.GlobalBestparam)))
        self.comm.Bcast([self.GlobalBestscore   ,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.GlobalBesteffparam,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.GlobalBestparam   ,MPI.DOUBLE], root = 0)

    # finished
    def updateFather(self):
        """NCS 16 Update father with child."""
        father_f = self.father_reward[self.id] - self.GlobalBestscore + self.eps
        child_f = self.child_reward[self.id] - self.GlobalBestscore + self.eps

        child_f = child_f / (child_f + father_f)
        if child_f / (self.corr_new + self.eps) < self.llambda:
            self.updatecount += 1
            self.eff_params = self.eff_params_new_all[self.id].copy()
            self.eff_params_new_all = np.zeros(
                (self.cpus, self.d)
            )
            self.father_reward[self.id] = self.child_reward[self.id]
        self.comm.Allgather(
            [self.eff_params    , MPI.DOUBLE],
            [self.eff_params_all, MPI.DOUBLE]
        )

    def updateSigma(self):
        """NCS 21-22 Update sigma"""
        self.sigma[self.updatecount / self.epoch < 0.2] = (
        	self.sigma[self.updatecount / self.epoch < 0.2] * self.r
        )
        self.sigma[self.updatecount / self.epoch > 0.2] = (
             	self.sigma[self.updatecount / self.epoch > 0.2] / self.r
        )


    # tested
    def getRandomMatrix(self,seed):
        """Generate a pesudo random matrix with seed.
        Return a pesudo matrix [D,d] related with seed.
        """
        rng = np.random.RandomState(seed)
        A = rng.standard_normal((self.D, self.d),dtype=np.double)
        return A

    def getRandomVector(self,ini_seed,idx,s):
        """Generate a [s,d] random vector with ini_seed.
        Ini_seed is a grandpa seed to generate a [D] seed list."""
        rng = default_rng(ini_seed)
        child_seed = rng.integers(0,999999,size = (self.D,))
        rng = default_rng(child_seed[idx])
        x = rng.standard_normal((s,self.d))
        return x

    # tested
    def getSeed(self):
        """Get seeds list(seeds in a group is same).
        E.g.[232913,232913,232913,345676,345676,345676,894356,894356,894356]
        """
        seeds = np.zeros((self.cpus,),dtype = 'i')
        rng = np.random.RandomState(int(time.time()))
        for i in range(self.mu):
            s = rng.randint(999999)
            seed = [s] * self.lam
            start = i * self.lam + 1
            seeds[start:start + self.lam] = seed
        return seeds

    # TODO: analyze time in different scale
    def from_x_to_y(self, x):
        """Map effective params x (D-dimension) to y (d-dimension).
            v1: y = (A-1) * x  (outdate)
            v2: don't need x to y
        """
        # y = np.linalg.solve(self.A[self.group_id], x)
        A = self.getRandomMatrix(self.seeds[self.id])
        if self.group_id is not None:
            y = np.dot(np.linalg.pinv(A, x))
        else:
            y = np.empty((self.d,))
        return y  # y

    def from_y_to_x(self, y):
        """Map effective params y (d-dimension) to x (D-dimension).
            v1: x = A * y
            v2: x' = ax + A*y
            v3: x'[j] = ax + A[j]*y
        """
        tmp = np.empty((self.D,))
        s = self.slice_len
        assert s <= self.D
        if self.group_id is not None:
            for j in range(0,self.D,s):
                if j + s > self.D:
                    s = self.D - 1 - j
                tmp[j:j+s,] = np.dot(self.getRandomVector(self.seeds[self.id],j,s),y)
            x = self.alpha * self.param + tmp
        else:
            x = tmp
        return x

    def rollout(self, parameters, test=False):
        """对策略中的 rollout 的封装，支持 k 次评估, parameters 为 D 维，对应Line8
        返回值：
            msg: [mean_reward, sum_len]
                第一个为平均得分
                第二个为消耗的训练帧总和
        """
        s = time.time()
        lens = [0]
        rews = [0]
        e_r = 0
        e_l = 0
        self.policy.set_parameters(parameters)
        if test is not True:
<<<<<<< HEAD
            k = self.k
            env_seed = np.random.choice(self.train_set)
        else:
            env_seed = np.random.choice(self.test_set)
            k = 5
=======
            k = int(np.ceil(self.k *(1 + self.steps_passed / self.steps_max)))
            env_seed = np.random.choice(self.train_set)
        else:
            env_seed = np.random.choice(self.test_set)
            k = 1
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        for j in range(k):
            e_rew, e_len = self.policy.rollout(env_seed)
            e_r += e_rew
            e_l += e_len
        rews[0] = e_r / k
        lens[0] = e_l
        msg = np.array(rews + lens)
        # self.logger.log_for_debug("Update Rollout 501 %d %s" % (self.id,rews[0]))
        return msg

    def retest(self):
        """Test GlobalBestparams for 30 times.
        Result will be save in log_retest dict."""
        if self.id == 0:
            if self.steps_passed - self.last_retest_steps > 1000000:
                self.last_retest_steps = self.steps_passed
                self.logger.log_for_debug("start evaluation on id")
                rew = 0
                for i in range(30):
                    r, _ = self.rollout(self.GlobalBestparam, test=True)
                    rew += r
                rew = rew/30
                self.log_retest["steps"].append(self.steps_passed)
                self.log_retest["performance"].append(rew)
                self.logger.log("------------------------------------")
                self.logger.log("Step".ljust(25) + "%s" % str(self.steps_passed))
                self.logger.log("Performance".ljust(25) + "%s" % str(rew))
                self.logger.log("------------------------------------")
                print("Step".ljust(25) + "%s" % str(self.steps_passed))
                print("Performance".ljust(25) + "%s" % str(rew))

    @staticmethod
    def calBdistance(param1, param2, sigma1, sigma2):
        """计算分布之间的距离
        参数：
            param1(np.ndarray): 分布1的均值
            sigma1(np.ndarray): 分布1的协方差

            param2(np.ndarray): 分布2的均值
            sigma2(np.ndarray): 分布2的协方差

        返回值：
            分布之间的距离值
        """
        xi_xj = param1 - param2
        big_sigma1 = sigma1 * sigma1
        big_sigma2 = sigma2 * sigma2
        big_sigma = (big_sigma1 + big_sigma2) / 2
        part1 = 1 / 8 * np.sum(xi_xj * xi_xj / (big_sigma + 1e-8))
        part2 = (
            np.sum(np.log(big_sigma + 1e-8))
            - 1 / 2 * np.sum(np.log(big_sigma1 + 1e-8))
            - 1 / 2 * np.sum(np.log(big_sigma2 + 1e-8))
        )
        return part1 + 1 / (2 * part2 + 1e-8)

    def calCorr(self, params_list, param, sigma_all, sigma):
        """计算分布param的相关性
        参数：
            n(int): the number of parameters

            param(np.ndarray): 当前分布的均值
            sigma(np.ndarray): 当前分布的协方差

            param_list(np.ndarray): 所有分布的均值
            sigma_all(np.ndarray): 所有分布的协方差

            rank(int): 当前线程的id
            group_id(int): 当前分组的id
        返回值：
            这个分布的相关性
        """
        start = self.group_id * self.group_cpus + 1
        end = (self.group_id + 1) * self.group_cpus
        DBlist = []
        for i in range(start, end):
            # i 是该进程在所有进程中序号
            if i != self.id:
                param2 = params_list[i]
                sigma2 = sigma_all[i]
                DB = self.calBdistance(param, param2, sigma, sigma2)
                DBlist.append(DB)
        return np.min(DBlist)

    def calDist(self,params):
        mean_ = np.around(np.mean(params),decimals=4)
        min_ = np.around(np.min(params),decimals=4)
        max_ = np.around(np.max(params),decimals=4)
        sigma = np.around(np.var(params),decimals=4)
        return min_,max_,mean_,sigma


    def syncOneValue(self, v):
        """工具函数，用于同步每个线程的单个标量值到每个线程的向量中。
        对mpi的Allgather的简单封装

        Args：
            v：标量
        Return：
            np.ndarray ： 大小为 cpus
        """
        v_t = np.array([v],dtype= 'd')
        v_all = np.zeros((self.cpus,),dtype= 'd')
        self.comm.Allgather([v_t, MPI.DOUBLE], [v_all, MPI.DOUBLE])
        return v_all.flatten()

    def syncOneVector(self, v):
        v_t = np.array(v)
        v_all = np.zeros((self.cpus, v_t.shape[0]))
        self.comm.Allgather([v_t, MPI.DOUBLE], [v_all, MPI.DOUBLE])
        return v_all

    # ------------------log-----------------------
    def logNCS(self, iter_start_time, cost_steps):
        """log after ncs iteration"""
        logger = self.logger
        self.comm.Allgather([self.Bestscore        ,MPI.DOUBLE],
                            [self.Bestscore_all    ,MPI.DOUBLE])
        if self.id == 0:
            iteration_time = time.time() - iter_start_time
            time_elapsed = (time.time() - self.start_time) / 60
            logger.log("------------------------------------")
            logger.log("Iteration".ljust(25) + "%d" % self.iter)
            logger.log("Phase".ljust(25) + "%d" % self.phase)
            logger.log("Child Reward".ljust(25) + "%s" % str(np.around(self.child_reward.flatten(),decimals=2)))
            logger.log("Bestscore in all process".ljust(25) + "%s" % str(np.around(self.Bestscore_all.flatten(),decimals=2)))
            logger.log("StepsThisIter".ljust(25) + "%f" % cost_steps)
            logger.log("StepsSinceStart".ljust(25) + "%f" % self.steps_passed)
            logger.log("IterationTime".ljust(25) + "%f" % iteration_time)
            logger.log("TimeSinceStart".ljust(25) + "%f" % time_elapsed)

    def logRE(self,phase_start_time,cost_steps):
        logger = self.logger
        msg = self.rollout(self.GlobalBestparam, test=True)
        logger.save_parameters(self.GlobalBestparam, self.phase)
        rew = msg[0]
        test_rewards = self.syncOneValue(rew)
        testrew = np.mean(test_rewards[:30])
        if self.id == 0:

            # msg = self.rollout(self.GlobalBestparam, test=True)
            # rew = msg[0]

            self.log_retest["steps"].append(self.steps_passed)
            self.log_retest["performance"].append(testrew)
            iteration_time = time.time() - phase_start_time
            time_elapsed = (time.time() - self.start_time) / 60
            logger.log("------------------------------------")
            logger.log("Phase".ljust(25) + "%d" % self.phase)
            logger.log(
                "Local Bestscore:".ljust(25) + "%s" % str(self.LocalBestscores.flatten())
            )
            logger.log(
                "Global Bestscore:".ljust(25) + "%s" % str(self.GlobalBestscore)
            )
            logger.log(
                "Retest Bestscore:".ljust(25) + "%s" % str(testrew)
            )
            seed = [self.seeds[i] for i in range(0,self.cpus,self.lam)]
            logger.log("Random embedding seed".ljust(25) + "%s" % (str(seed)))
            logger.log("StepsThisPhase".ljust(25) + "%f" % cost_steps)
            logger.log("StepsSinceStart".ljust(25) + "%f" % self.steps_passed)
            logger.log("PhaseTime".ljust(25) + "%f" % iteration_time)
            logger.log("TimeSinceStart".ljust(25) + "%f" % time_elapsed)

    def logBasic(self):
        logger = self.logger
        if self.id == 0:
            logger.log("----------------NCSRE---------------")
            logger.log("High dimension space %d" % self.D)
            logger.log("Low dimension space %d" % self.d)
            logger.log("Number of NCS ind(lambda) %d" % self.lam)
            logger.log("Number of NCS pop(mu) %d" % self.mu)
            logger.log("Random embedding phases %d" % self.episode)
            logger.log("Timestep limit %d" % self.steps_max)
            logger.log("Group cpus %d" % self.group_cpus)
            logger.log("Total cpus %d" % self.cpus)
            logger.log("Sigma reupdated every epoch %d" % self.sigma_reup)
            logger.log("-----------------NCS---------------")
            logger.log("Evaluate times %d" % self.k)
            logger.log("Epoch %d" % self.epoch)
            logger.log("Llambda(success ratio)%d" % self.llambda)
            logger.log("Sigma init %.3f" % self.sigma0)
            logger.log("Sigma update ratio %.2f" % self.r)
            logger.log("-----------------Other---------------")
            logger.log("Len of phase %d" % self.len_phase)
            logger.log("Random seed %d" % self.random_seed)
            logger.log("Slice num of matrix %d" % self.slice_len)

    def logPath(self):
        """return log path as logs_mpi/Alien/mu5/lam5/dimension10/run"""
        if self.args["env_type"] == "atari":
<<<<<<< HEAD
            logpath = "logs_mpi/%s/NCSRE/mu%d/lam%s/dim%d/%s" % (
=======
            logpath = "logs_mpi/%s/%s/mu%d/lam%s/dim%d/%s" % (
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
                self.args["game"],
                self.algoname,
                self.mu,
                self.lam,
                self.d,
                self.args["run_name"],
            )
        elif self.args["env_type"] == "function":
<<<<<<< HEAD
            logpath = "logs_mpi/func%s/NCSRE/mu%d/lam%s/dim%d/%s" % (
                self.args["func_id"],
=======
            logpath = "logs_mpi/func%s/%s/mu%d/lam%s/dim%d/%s" % (
                self.args["func_id"],
                self.algoname,
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
                self.mu,
                self.lam,
                self.d,
                self.args["run_name"],
            )
        return logpath

    def saveDetail(self):
        """Save detailed log for debug as pickle.
        1. Correlation factor of each individual in each gen.
        2. Sigma updated.
        3. Successful rate.
        """
        filepath = os.path.join(self.logPath(), str(self.id) + "detail.pickle")
        with open(filepath, "wb") as f:
            pickle.dump(self.log_detail, f)

    def saveRetestLog(self):
        """Save log for retest"""
        filepath = os.path.join(self.logPath(), "retest_log.pickle")
        with open(filepath, "wb") as f:
            pickle.dump(self.log_retest, f)

    def bestTest(self,repeat=200):
        """运行算法之后，对最好的个体进行测试200次，并保存结果
        """
        # logger = self.logger
        # print("save And Test best")
        # test best
        self.k = 1
        self.logger.save_parameters(self.GlobalBestparam,self.phase)
        # 计算每个线程跑多少次游戏
        test_times = int(repeat / self.cpus)
        # 剩余的给主线程来跑
        reminder = repeat - test_times * self.cpus
        final_rews = []
        for i in range(test_times):
            msg = self.rollout(self.GlobalBestparam,test=True)
            final_rews.append(msg[0])
        part_one_reward = self.syncOneValue(np.mean(final_rews))

        if self.id == 0:
            part_two_reward = []
            for i in range(reminder):
                msg = self.rollout(self.GlobalBestparam, test=True)
                part_two_reward.append(msg[0])
            self.final_eval = (np.sum(part_two_reward) + np.sum(part_one_reward) * test_times) / repeat
            self.logger.log("Globalbest solution last test result: %s"%str(self.final_eval))
            self.saveLog('log_retest',self.log_retest)
            self.saveLog('log_train',self.log_train)

<<<<<<< HEAD
=======
            with open("log.csv",'w') as f:
                f.write(str(self.args["run_name"])+',NCSRE,'+str(self.args["game"]+','+str(self.final_eval)+','+str(self.GlobalBestscore)))

    def saveLog(self,filename,log):
        """保存重新测试的日志
        """
        filepath = os.path.join(self.logPath(), filename + '.pickle')
        with open(filepath, 'wb') as f:
            pickle.dump(log, f)

    def log_traincurve(self):
        """save train curve png"""
        self.logger.draw_single(
            self.log_retest,
            self.algoname,
            self.args["game"],
            "train-curve"
        )
        #self.logger.draw_single(
        #    self.log_train,
        #    self.algoname,
        #    self.args["game"],
        #    "test-curve"
        #)
        self.logger.draw_two(
            self.log_retest,
            self.log_train,
            self.algoname,
            self.args["game"],
            "train-test-curve"
        )

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
# discard
def args_parser(args):
    import argparse
    parser = argparse.ArgumentParser(description="args for NCSRE")
    # parser.add_argument("-n","--ncpu", default=9, type=int, help="num of cpus")
    parser.add_argument("-r","--run_name", default="debug1", type=str, help="run name")
    # parser.add_argument("--env_type", default="atari", type=str, help="test env type(function or atari)")
    parser.add_argument("-g","--game", default="SpaceInvaders", type=str, help="game in atari")
    # parser.add_argument("-f","--func_id", default=1, type=int, help="function id in benchmark(1~7)")
    parser.add_argument("-d", default=100, type=int, help="effective dimensions")
    parser.add_argument("--len", default=20, type=int, help="max times of random embedding")
    parser.add_argument("-l","--lam", default=8, type=int, help="number of NCS individuals in a subprocess")
    parser.add_argument("--mu", default=3, type=int, help="number of NCS populations")
    parser.add_argument("-e","--eva_times", default=10, type=int, help="evaluation repeated times")
    parser.add_argument("--sigma_reup", default=0, type=bool, help="=0 sigma start from 0 each phase =1")
    kwargs = parser.parse_args()
    args['run_name'] = kwargs.run_name
    args['game'] = kwargs.game
    args['d'] = kwargs.d
    # args['m'] = kwargs.m
    args['lam'] = kwargs.lam
    args['len_phase'] = kwargs.len
    args['mu'] = kwargs.mu
    args['eva_times'] = kwargs.eva_times
    args['sigma_reup'] = kwargs.sigma_reup
    return args


@click.command()
@click.option('--run_name', '-r', required=True, default = 'debug',type=click.STRING, help='Name of the run, used to create log folder name')
@click.option('--env_type', '-e', required=True, default = 'atari',type=click.Choice(['atari', 'function']))
@click.option('--game', '-g', type=click.STRING, default = 'Alien', help='game name for RL')
@click.option('--func_id', '-f', type=click.INT, help='function id for function optimization')
@click.option('--func_dim', '-d', type=click.INT, help='the dimension of solution for function optimization')
@click.option('--config','-c', default='./config/NCSoptimal.json', help='configuration file path')
def main(run_name, env_type, game, func_id, func_dim, config):
    with open(config, 'r') as f:
        kwargs = json.loads(f.read())
    kwargs['run_name'] = run_name
    kwargs['env_type'] = env_type
    kwargs['game'] = game
    kwargs['func_id'] = func_id
    kwargs['func_dim'] = func_dim
    algo = NCSREAlgo(kwargs)
    algo.run()
    algo.bestTest()
<<<<<<< HEAD

if __name__ == "__main__":
    main()
    
    
=======
    #algo.log_traincurve()

if __name__ == "__main__":
    main()


>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
