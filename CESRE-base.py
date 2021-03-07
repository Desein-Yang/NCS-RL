#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CESRE-base
@Time    :   2020/06/27 16:41:31
@Author  :   Qi Yang
@Describtion:  CESRE-base 算法 (调用base类的)
'''


# here put the import lib
import numpy as np
import gym
import time
import os
import pickle
import warnings
import cProfile
import click
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from mpi4py import MPI
from src.logger import Logger
from src.policy import Policy,FuncPolicy
from src.testfunc import test_func_bound
# from memory_profiler import profile
# from memory_profiler import memory_usage
from src.base import BaseAlgo

class REAlgo(BaseAlgo):
    def __init__(self, **kwargs):
        super(BaseAlgo,self).__init__(**kwargs)
<<<<<<< HEAD
        
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        # mpi
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.cpus = self.comm.Get_size()

        # Re Hyper
        self.args = kwargs
        # self.k = self.args['k']
        # self.lr = self.args['learning_rate']
        # self.sigma0 = self.args['sigma0']
        self.algoname = 'RE'
<<<<<<< HEAD
        # 默认子算法为 CES 使用其他子算法改写 optimization 
=======
        # 默认子算法为 CES 使用其他子算法改写 optimization
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        self.subalgoname = 'CES'
        self.logger = Logger(self.logPath())
        # self.steps_max = args["steps_max"]

        # RE 特有超参数
        self.D = self.n  # model size in base
        self.d = args["effdim"]
<<<<<<< HEAD
        self.lam = args["lam"]  
        self.mu = args["mu"]  
=======
        self.lam = args["lam"]
        self.mu = args["mu"]
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        # self.episode = args["m"]  # m : number of RE phase
        self.len_phase = args["len_phase"]
        # self.group_cpus = args["group_cpus"]
        # 区别于 其他算法的 lam 直接为 cpus - 1
        # 此处 Lam 为一个种群个体数

        # Re 特有分组
        # rank 表示该CPU在所有CPU中编号 0 1 2 - cpus（与Base一致）
        # id 表示CPU在该组CPU中编号 0 1 2 - lam-1
        # group_id 表示该CPU所在组的编号 0 1 2 - mu-1
        self.group_cpus = self.lam
        if self.rank != 0:
            self.id = (self.rank - 1) % self.group_cpus
            self.group_id = int(np.floor((self.rank - 1) / self.group_cpus))
        else:
            self.id = 0
            self.group_id = self.mu
<<<<<<< HEAD
        
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        # Re 特有随机矩阵生成
        self.seeds = self.getSeed()
        self.slice_len = 10000

        # Re 特有 withdraw factor
        self.alpha = np.ones(1) * args['alpha0']
        self.alpha_all = np.empty((self.cpus, 1))
        self.comm.Allgather([self.alpha, MPI.DOUBLE],
                            [self.alpha_all,MPI.DOUBLE])

        # d 空间搜索范围和 D_ 空间搜索范围
        self.L , self.H = (-0.1,0.1)
        self.L_, self.H_ = (-5,5)

        # if self.args['seed'] is None:
        #     self.randomSeed = np.random.randint(100000)
        # else:
        #     self.randomSeed = self.args['seed']
        # self.rs = np.random.RandomState(self.randomSeed)
        # self.logger = Logger(self.logPath())
<<<<<<< HEAD
        
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        # 创建策略模型以及设置对应的超参数
        # if self.args['env_type'] == 'atari':
        #     env = gym.make("%sNoFrameskip-v4" % self.args['game'])
        #     env = wrap_dqn(env)
        #     self.policy = Policy(env, network=self.args['network'], nonlin_name=self.args['nonlin_name'])
        #     self.steps_max = self.args['steps_max']
        # elif self.args['env_type'] == 'function':
        #     self.policy = FuncPolicy(self.args['D'], self.args['function_id'], self.rank, self.randomSeed)
        #     self.steps_max = 5000 * self.args['D']
        #     bound = test_func_bound[self.args['function_id']]
        #     self.sigma0 = (bound[1] - bound[0]) / (self.cpus - 1)

        # 同步不同线程的参数到param_all变量中
        # self.param = self.policy.get_parameters()
        # self.n = len(self.param)
        # self.param_all = np.empty((self.cpus, self.n))
        # self.comm.Allgather([self.param, MPI.DOUBLE], [self.param_all, MPI.DOUBLE])
<<<<<<< HEAD
    
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

        # self.param_new = np.zeros(self.n)
        # self.sigma = np.ones(self.n) * self.sigma0
        # self.sigma_all = np.ones((self.cpus, self.n))
        # self.BestParam_t = self.param.copy()
        # self.BestParam_t_all = self.param_all.copy()
        # self.BestScore_t = np.zeros(1)
        # self.BestScore_t_all = np.zeros((self.cpus, 1))
        # Bestscore -> BestScore_t
<<<<<<< HEAD
        # Bestscore_all --> BestScore_t_all  
              
=======
        # Bestscore_all --> BestScore_t_all

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        # RE 特有 小空间优化
        # param is for rank ! = 0
        # param_all is for all
        self.eff_params = np.random.uniform(self.L,self.H,(self.d,))
        self.logger.log_for_debug('self.init 118 eff_params ' + str(self.calDist(self.eff_params)))
        self.eff_params_all = np.zeros((self.cpus,self.d))
        self.comm.Allgather(
            [self.eff_params, MPI.DOUBLE],
            [self.eff_params_all, MPI.DOUBLE]
        )
        self.eff_params_new = self.eff_params.copy()
        self.eff_params_new_all = self.eff_params_all.copy()


        # 废弃不用 base 中的参数
        # BEST in all threads
        # self.BESTSCORE = 0
        # self.BESTSCORE_id = 0
        # self.BESTParam = np.empty(self.n)
        # GlobalbestScore --> BESTSCORE
        # BESTSCORE_id don't need
        # GlobalBestparam --> BESTParam
        # GlobalBesteffparam --> BESTeffParam

        # Re 特有
        self.BestEffparam = self.eff_params.copy()
        self.BestEffparam_all = self.eff_params_all.copy()

        # NCSRE line 10 , best local score and best params x
        self.LocalBestscores = np.zeros((self.lam, 1))
        self.LocalBestparams = np.random.uniform(self.L_,self.H_,(self.lam, self.D,))
        self.LocalBesteffparams = np.random.uniform(self.L,self.H,(self.lam, self.d,))

        # NCSRE line 12 , best params x in history and all local subprocess
        # self.GlobalBestscore = np.zeros(1)
        # self.GlobalBestparam = np.zeros((self.D,))
        self.BESTeffParam = np.zeros((self.d,))

        # 通用初始化
<<<<<<< HEAD
        self.eps = 1e-8 
=======
        self.eps = 1e-8
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        self.steps_passed = 0
        self.last_retest_steps = 0
        self.iter = 1
        self.phase = 1

        self.log_retest = {
            "step": [],
            "performance": []
        }
        self.log_detail = {
            "rank" : self.rank,
            "corr_new": [],
            "corr_old": [],
            "eff_params": [], # [(max,min,mean,sigma),]
            "params": [],
            "sigma": [],
            "succ" : [],
            "lambda":[]
        }

        self.logBasic()
        self.firstEvaluation()
        self.reward_child = np.zeros((self.cpus,),dtype='d')

    # random embedding math utils
    # TODO: analyze time in different scale
    def embed(self, x):
<<<<<<< HEAD
        """Map effective params x (D-dimension) to y (d-dimension).  
=======
        """Map effective params x (D-dimension) to y (d-dimension).
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
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

    def expand(self, y):
<<<<<<< HEAD
        """Map effective params y (d-dimension) to x (D-dimension).  
            v1: x = A * y  
            v2: x' = ax + A*y
            v3: x'[j] = ax + A[j]*y 
=======
        """Map effective params y (d-dimension) to x (D-dimension).
            v1: x = A * y
            v2: x' = ax + A*y
            v3: x'[j] = ax + A[j]*y
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        """
        tmp = np.empty((self.D,))
        s = self.slice_len
        if self.group_id is not None:
            for j in range(0,self.D,s):
                if j + s > self.D:
                    s = self.D - 1 - j
                tmp[j:j+s,] = np.dot(self.getRandomVector(self.seeds[self.id],j,s),y)
            x = self.alpha * self.param + tmp
        else:
            x = tmp
        return x

    def firstEvaluation(self):
        """evaluation for first initialization
        """
        self.initVbn()
        msg = self.rollout(self.param)
        reward_child = msg[0]
        # results = np.empty((self.cpus, 2))
        self.reward_child = self.syncOneValue(reward_child)
        # TODO 对应bestparam 基类和子类变量名
<<<<<<< HEAD
        self.updateBest_t()     
=======
        self.updateBest_t()
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        self.updateLocalbest()
        self.updateGlobalbest()
        if self.rank == 0:
            self.seeds = self.getSeed()
        else:
            self.seeds = np.empty((self.cpus,),dtype = 'i')
        self.comm.Bcast([self.seeds, MPI.INT],root = 0)

    def run(self):
        """Re 主体部分框架"""
        self.start_time = time.time()

        while self.steps_passed <= self.steps_max:
            phase_start_time = time.time()

            # 开启重新 sigma 机制
            # self.updateSigma()

            # 分发随机种子
            if self.rank == 0:
                self.seeds = self.getSeed()
            else:
                self.seeds = np.empty((self.cpus,),dtype = 'i')
            self.comm.Bcast([self.seeds, MPI.INT],root = 0)
            assert self.seeds is not None, 'Seed set Error'

            # TODO：更新父代 updateFather中实现
            self.param = self.LocalBestparams[self.id].copy()
            self.eff_params = self.LocalBesteffparams[self.id].copy()

            # 主体算法实现 CES，NCS 等
            phase_cost_steps = self.optimize()  # best params has been synced

            # Notes: now we use only 1 best will be pick
            # So update global best is empty and pass
            self.updateGlobalbest()

            self.logRE(phase_start_time,phase_cost_steps)
            self.phase += 1
            self.sampleBest(self.steps_passed, interval=490000)
        self.saveDetail()
        self.saveRetestLog()

    def genChild(self):
<<<<<<< HEAD
        """改写为在小空间上加噪声  
=======
        """改写为在小空间上加噪声
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        然后转换到大空间"""
        self.eff_params_new = self.eff_params + self.rs.normal(scale = self.sigma,size = self.d)
        self.eff_params_new = np.clip(self.eff_params_new,self.L,self.H)
        self.logger.log_for_debug('genchild %s' % str(self.calDist(self.eff_params_new)))
        assert eff_params_new.shape == (self.d,), 'Eff_param_new shape Error'
<<<<<<< HEAD
           
        self.param_new = self.expand(self.eff_params_new)
        self.param_new = np.clip(self.param_new,self.L_,self.H_)
            
=======

        self.param_new = self.expand(self.eff_params_new)
        self.param_new = np.clip(self.param_new,self.L_,self.H_)

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
    def evalChild(self):
        """改写为由小空间升维的子代进行评估
        在产生子代中和更新最佳参数中修改，此处和base类保持一致"""
        pass

    def optimize(self):
        """单个算法主体函数"""
        phase_cost_steps = 0
        while self.iter <= self.phase * self.len_phase:
            iter_start_time = time.time()
            steps_this_iter = 0

            reward_child, cost_steps = self.evalChild()
            self.reward_child = reward_child.copy()
<<<<<<< HEAD
            
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
            # sync here
            self.updateLocalbest()
            self.updateFather()
            steps_this_iter += cost_steps
            steps_passed += cost_steps
            phase_cost_steps += cost_steps

            self.logSub(iter_start_time, steps_this_iter, steps_passed)
            self.iteration += 1
        return phase_cost_steps

<<<<<<< HEAD
    def updateFather(self):
        """ 更新参数 """
        if self.rank != 0:
            w = self.calweight()
            for i in range(self.lam):
                self.eff_params = self.eff_params + self.lr * w[i] * (self.eff_params_new - self.eff_params)
=======
    def updateFather(self,group_id):
        """ 更新当前分组的参数 """
        if self.rank != 0:
            w = self.calweight(group_id)
            eff_params_new = self.syncOneVector(self.eff_params_new)

            for i in range(self.mu):
                self.eff_params = self.eff_params + self.lr * w[i] * (eff_params_new[group_id*self,mu] - self.eff_params)
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
                self.updateLr()
        self.eff_params_all = self.syncOneVector(self.eff_params)

    # weight calculate
<<<<<<< HEAD
    def calweight(self):
        rank = np.argsort(self.reward_child[1:]) + 1 #返回每个子代的编号,rank0 不包含在内
        tmp = [np.log(self.lam+0.5)-np.log(j) for j in rank]
=======
    def calweight(self,group_id):
        """仅仅返回当前分组的权重"""
        rank = np.argsort(self.reward_child[group_id*self.mu+1:(group_id+1)*self.mu]) + 1 #返回每个子代的编号,rank0 不包含在内
        tmp = [np.log(self.mu+0.5)-np.log(j) for j in rank] # 其他算法的mu是这里的lam
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        weight = tmp / np.sum(tmp)
        return weight

    def updateSigma(self):
<<<<<<< HEAD
        """更新sigma.  
        分为继承机制和重新从1开始机制  
=======
        """更新sigma.
        分为继承机制和重新从1开始机制
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        重新开始机制则启动此函数"""
        self.sigma = np.ones(self.d, dtype = np.double) * self.sigma0
        self.comm.Allgather([self.sigma, MPI.DOUBLE],
                            [self.sigma_all, MPI.DOUBLE])

    def updateLr(self):
        """学习率0.99衰减"""
        self.lr = self.lr * 0.99

    def updateBEST(self):
<<<<<<< HEAD
        """重写了base类   
=======
        """重写了base类
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        update local bestfound of each x"""
        pass
        # 本类被local best取代

    def updateBest_t(self, score, param):
<<<<<<< HEAD
        """更新当前线程的最优个体得分与参数  
=======
        """更新当前线程的最优个体得分与参数
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        增加了子空间的保存
        """
        if score > self.BestScore_t[0]:
            self.BestScore_t[0] = score
            self.BestParam_t = param.copy()
            self.BestEffparams = self.eff_params_new.copy()

    def updateLocalbest(self):
<<<<<<< HEAD
        """ 更新所有相同投影子代中最好的局部最优/相同子代的不同投影最好  
        目前采用第一种
        
        v1: return one bestfound in all groups.  
        v2: return lambda bestfound in each random matrix.  
        now I use v2 and it is global.    
=======
        """ 更新所有相同投影子代中最好的局部最优/相同子代的不同投影最好
        目前采用第一种

        v1: return one bestfound in all groups.
        v2: return lambda bestfound in each random matrix.
        now I use v2 and it is global.
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

        """
        self.comm.Allgather([self.BestScore_t        ,MPI.DOUBLE],
                            [self.BestScore_t_all    ,MPI.DOUBLE])
        self.comm.Allgather([self.BestEffparam    ,MPI.DOUBLE],
                            [self.BestEffparam_all,MPI.DOUBLE])

        if self.rank == 0:
            self.logger.log_for_debug(str(self.BestScore_t_all))
            for i,score in enumerate(self.BestScore_t_all[1:]):
                # Notes: cpu 0 is not included
<<<<<<< HEAD
                # has been test 
=======
                # has been test
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
                rank = i % self.group_cpus
                if score > self.LocalBestscores[rank]:
                    self.logger.log_for_debug('Before update local best eff all'+str(self.calDist(self.LocalBesteffparams[rank])))
                    self.LocalBestscores[rank] = score
                    self.LocalBesteffparams[rank] = self.BestEffparam_all[i+1].copy()
                    self.LocalBestparams[rank] = np.clip(self.from_y_to_x(self.LocalBesteffparams[rank]),self.L_,self.H_)

                    self.logger.log_for_debug("Update Local Best 417 %d %s" % (self.id,self.LocalBestscores))
                    self.logger.log_for_debug('After update local best eff all'+str(self.calDist(self.LocalBesteffparams[rank])))

        self.comm.Bcast([self.LocalBestscores   ,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.LocalBesteffparams,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.LocalBestparams   ,MPI.DOUBLE], root = 0)
<<<<<<< HEAD
            
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
    def updateGlobalbest(self):
        """同步全局NCS的BESTFOUND"""
        if self.rank == 0:
            idx = np.argmax(self.LocalBestscores.flatten())
            self.BESTSCORE = np.max(self.LocalBestscores.flatten())
            self.BESTParam = self.LocalBestparams[idx].copy()
            self.logger.log_for_debug("best id"+str(idx))
            self.logger.log_for_debug("Update Global Best 416" + str(self.calDist(self.BESTParam)))
<<<<<<< HEAD
            
        self.comm.Bcast([self.BESTSCORE   ,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.BESTeffParam,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.BESTParam   ,MPI.DOUBLE], root = 0) 
        if self.rank != 0:
            self.logger.log_for_debug('update 423')
           
=======

        self.comm.Bcast([self.BESTSCORE   ,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.BESTeffParam,MPI.DOUBLE], root = 0)
        self.comm.Bcast([self.BESTParam   ,MPI.DOUBLE], root = 0)
        if self.rank != 0:
            self.logger.log_for_debug('update 423')

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1


    # tested
    def getRandomMatrix(self,seed):
<<<<<<< HEAD
        """Generate a pesudo random matrix with seed.  
        Return a pesudo matrix [D,d] related with seed.  
=======
        """Generate a pesudo random matrix with seed.
        Return a pesudo matrix [D,d] related with seed.
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        """
        rng = np.random.RandomState(seed)
        A = rng.standard_normal((self.D, self.d),dtype=np.double)
        return A
<<<<<<< HEAD
    
    def getRandomVector(self,ini_seed,idx,s):
        """Generate a [s,d] random vector with ini_seed.  
=======

    def getRandomVector(self,ini_seed,idx,s):
        """Generate a [s,d] random vector with ini_seed.
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        Ini_seed is a grandpa seed to generate a [D] seed list."""
        rng = default_rng(ini_seed)
        child_seed = rng.integers(0,999999,size = (self.D,))
        rng = default_rng(child_seed[idx])
        x = rng.standard_normal((s,self.d))
        return x

    # tested
    def getSeed(self):
<<<<<<< HEAD
        """Get seeds list(seeds in a group is same).   
        origin seed is from base.py randomSeed.    
        E.g.[232913,232913,232913,345676,345676,345676,894356,894356,894356] 
=======
        """Get seeds list(seeds in a group is same).
        origin seed is from base.py randomSeed.
        E.g.[232913,232913,232913,345676,345676,345676,894356,894356,894356]
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        """
        seeds = np.zeros((self.cpus,),dtype = 'i')
        rng = np.random.RandomState(self.randomSeed)
        for i in range(self.mu):
<<<<<<< HEAD
            s = rng.randint(999999) 
=======
            s = rng.randint(999999)
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
            seed = [s] * self.lam
            start = i * self.lam + 1
            seeds[start:start + self.lam] = seed
        del rng # 销毁减少内存占用
        return seeds

<<<<<<< HEAD
=======
    @staticmethod
    def syncLocalVector(group_id,x,source):
        """同步本组变量"""
        if self.rank == source:
            self.comm.Send([x,MPI.DOUBLE],dest = 0)# send to 0
        if self.rank == 0:
            self.comm.Recv([x,MPI.DOUBLE],dest = source)
            for i in range(group_id*self.mu+1,(group_id+1)*self.mu):
            self.comm.Send([x,MPI.DOUBLE],dest = i)
        if self.rank >= grou_id*self.mu+1 and self.rank <= (group_id+1)*self.mu:
            self.comm.Recv([x,MPI.DOUBLE],dest = 0)



>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
    # log utils
    def logRE(self,phase_start_time,cost_steps):
        logger = self.logger
        msg = self.rollout(self.BESTParam, test=True)
        rew = msg[0]
        test_rewards = self.syncOneValue(rew)
        testrew = np.mean(test_rewards[:30])
<<<<<<< HEAD
        if self.rank == 0: 
=======
        if self.rank == 0:
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
            self.log_retest["step"].append(self.steps_passed)
            self.log_retest["performance"].append(testrew)
            if self.args["env_type"] == "atari":
                iteration_time = time.time() - phase_start_time
                time_elapsed = (time.time() - self.start_time) / 60
                logger.log("------------------------------------")
                logger.log("Phase".ljust(25) + "%d" % self.phase)
                logger.log(
                    "Local Bestscore:".ljust(25) + "%s" % str(self.LocalBestscores.flatten())
                )
                logger.log(
                    "Global Bestscore:".ljust(25) + "%s" % str(self.BESTSCORE)
                )
                logger.log(
                    "Retest Bestscore:".ljust(25) + "%s" % str(rew)
                )
                seed = [self.seeds[i] for i in range(0,self.cpus,self.lam)]
                logger.log("Random embedding seed".ljust(25) + "%s" % (str(seed)))
                logger.log("StepsThisPhase".ljust(25) + "%f" % phase_cost_steps)
                logger.log("StepsSinceStart".ljust(25) + "%f" % self.steps_passed)
                logger.log("PhaseTime".ljust(25) + "%f" % iteration_time)
                logger.log("TimeSinceStart".ljust(25) + "%f" % time_elapsed)

    def logBasic(self):
        logger = self.logger
        if self.rank == 0:
            logger.log("----------------NCSRE---------------")
            logger.log("High dimension space %d" % self.D)
            logger.log("Low dimension space %d" % self.d)
            logger.log("Number of NCS ind(lambda) %d" % self.lam)
            logger.log("Number of NCS pop(mu) %d" % self.mu)
            logger.log("Random embedding phases %d" % self.episode)
            logger.log("Timestep limit %d" % self.steps_max)
            logger.log("Group cpus %d" % self.group_cpus)
            logger.log("Total cpus %d" % self.cpus)
            logger.log("Len of phase %d" % self.len_phase)
            logger.log("Random seed %d" % self.randomSeed)
            logger.log("Slice num of matrix %d" % self.slice_len)
            logger.log("Sigma reupdated every epoch %d" % self.sigma_reup)
            logger.log("Evaluate times %d" % self.k)
            self.logSubBasic()
<<<<<<< HEAD
    
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
    def logSubBasic(self):
        """输出子算法的基本信息"""
        logger = self.logger
        if self.rank == 0:
            logger.log("---------------%s-----------------" % self.subalgoname)
            logger.log("CES")


    def logSub(self,iter_start_time, steps_this_iter, steps_passed):
        """输出子算法的回合信息"""
        logger = self.logger
        if self.rank == 0:
            if self.args["env_type"] == "atari":
                iteration_time = time.time() - iter_start_time
                time_elapsed = (time.time() - self.start_time) / 60
                logger.log("------------------------------------")
                logger.log("Iteration".ljust(25) + "%d" % self.iter)
                logger.log("Phase".ljust(25) + "%d" % self.phase)
                logger.log("Child Reward".ljust(25) + "%s" % str(self.reward_child.flatten()))
                logger.log("Bestscore in all subprocess".ljust(25) + "%s" % str(self.BestScore_t_all.flatten()))
                logger.log("StepsThisIter".ljust(25) + "%f" % cost_steps)
                logger.log("StepsSinceStart".ljust(25) + "%f" % self.steps_passed)
                logger.log("IterationTime".ljust(25) + "%f" % iteration_time)
                logger.log("TimeSinceStart".ljust(25) + "%f" % time_elapsed)

    def logPath(self):
        """return log path as logs_mpi/Alien/NCSRE/mu5/lam5/dimension10/run"""
        if self.args["env_type"] == "atari":
            logpath = "logs_mpi/%s/%s/mu%d/lam%s/dimension%d/%s" % (
                self.args["game"],
                self.subalgoname + self.algoname,
                self.mu,
                self.lam,
                self.d,
                self.args["run_name"],
            )
        elif self.args["env_type"] == "function":
            logpath = "logs_mpi/function%s/%s/mu%d/lam%s/dimension%d/%s" % (
                self.args["function_id"],
                self.subalgoname + self.algoname,
                self.mu,
                self.lam,
                self.d,
                self.args["run_name"],
            )
        return logpath

    # debug util
    def saveDetail(self):
<<<<<<< HEAD
        """Save detailed log for debug as pickle.  
        1. Correlation factor of each individual in each gen.  
        2. Sigma updated.  
        3. Successful rate.    
=======
        """Save detailed log for debug as pickle.
        1. Correlation factor of each individual in each gen.
        2. Sigma updated.
        3. Successful rate.
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        """
        filepath = os.path.join(self.logPath(), str(self.rank) + "detail.pickle")
        with open(filepath, "wb") as f:
            pickle.dump(self.log_detail, f)

    def saveRetestLog(self):
        """Save log for retest"""
        filepath = os.path.join(self.logPath(), "retest_log.pickle")
        with open(filepath, "wb") as f:
            pickle.dump(self.log_retest, f)

def args_parser(args):
    import argparse
    parser = argparse.ArgumentParser(description="args for NCSRE")
    # parser.add_argument("-n","--ncpu", default=9, type=int, help="num of cpus")
    parser.add_argument("-r","--run_name", default="debug1", type=str, help="run name")
    # parser.add_argument("--env_type", default="atari", type=str, help="test env type(function or atari)")
    parser.add_argument("-g","--game", default="SpaceInvaders", type=str, help="game in atari")
    # parser.add_argument("-f","--function_id", default=1, type=int, help="function id in benchmark(1~7)")
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

# config and args dict
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
    "sigma_reup":1,

    # env
    "game": "Alien",
    "env_type": "atari",
    "function_id": 1,
    "run_name": "tuning-1",

    # ncs
    "epoch": 5,  # ncs epoch of update sigma
    "r": 0.2,  # success rate to update sigma
    "sigma0": 0.01,
    "lambda0": 1.0,
    "timesteps_limit": 1e8,

    # ncsre
    "D": 1000,  # model size need to be set in run
    "d": 10,
    "m": 100,  # m: number of max iter of RE phase
    "len_phase": 10,
    "mu": 3,  # # mu : number of NCS populations
    "lam": 4, # lambda : number of NCS individuals, = group_cpu if one ncs use one core
    "eva_times": 5,
    "alpha0": 1,
}
<<<<<<< HEAD
    
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

@click.command()
@click.option('--run_name', '-r', required=True, default = 'debug',type=click.STRING, help='Name of the run, used to create log folder name')
@click.option('--env_type', '-e', default = 'atari',type=click.Choice(['atari', 'function']))
@click.option('--game', '-g', type=click.STRING, default = 'Alien', help='game name for RL')
@click.option('--function_id', '-f', type=click.INT, help='function id for function optimization')
@click.option('--dimension', type=click.INT, help='the dimension of solution for function optimization')
@click.option('--lam', '-l', type=click.INT, default = 8, help='number of individuals in a subprocess')
@click.option('--mu', '-m', type=click.INT, default=1, help='the number of subprocesses')
@click.option('--effdim', '-d', type=click.INT, default=10, help='evaluation times in training')
@click.option('--phaselen','-p', type=click.INT, default=20, help='evaluation times in training')
@click.option('--sigma0', type=click.FLOAT, default=1.0, help='sigma0')
@click.option('--alpha0', type=click.FLOAT, default=1.0, help='alpha0')
@click.option('--evatimes', '-k', type=click.INT, default=10, help='evaluation times in training')
@click.option('--stepmax','-s', type=click.INT, default=25000000, help='maximum of framecount')
@click.option('--lr', type=click.FLOAT, default=1.0, help='init value of learning rate')
def main(run_name, env_type, game, function_id, dimension, lam,mu,effdim,phaselen,sigma0,alpha0, evatimes,stepmax,lr):
    # 算法入口
    kwargs = {
        'network': 'Nature',
        'nonlin_name': 'relu',
        'k': evatimes,
        'lam':lam,
        'mu':mu,
        'sigma0': sigma0,
        'run_name': run_name,
        'env_type': env_type,
        'game': game,
        'function_id': function_id,
        'D': dimension,
        'd': effdim,
        'phase_len':phaselen,
        'steps_max':stepmax,
        'learning_rate':lr,
    }
    algo = REAlgo(**kwargs)
    algo.run()
    # 最优解的最终测试以及保存日志
    algo.testBest()
<<<<<<< HEAD
    
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

if __name__ == '__main__':
    main()
