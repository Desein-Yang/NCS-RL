#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   NCSCC-base.py
@Time    :   2020/09/09 15:44:31
@Author  :   Qi Yang
@Version :   1.0
@Describtion:  NCSCC (use base class)
'''


# paper: 
import time
import gym
import pickle
import click
import os
import numpy as np
import json

from mpi4py import MPI

from src.policy import Policy,FuncPolicy
from src.logger import Logger
from src.testfunc import test_func_bound
from src.base import BaseAlgo,NCSAlgo
from src.decompose import Decomposer


# 以下三个变量用于描述版本号和具体内容之间的联系
# 如版本1，3 采用Bestfound_i 来填充
VersionUsingBestfound = [1, 3]
VersionUsingFather = [2, 4]
VersionDivideEveryEpoch = [3, 4]


class NCSCCAlgo(NCSAlgo):
    def __init__(self, **kwargs):
        '''算法类

        重要成员变量说明
            param     父代的参数（均值）
            param_all 所有父代的参数
            sigma     父代个体的协方差
            sigma_all 所有父代个体的协方差

            param_new 子代的参数

            BestParam_t     每个线程中的最优个体参数（分布均值）
            BestParam_t_all 所有线程中的最优个体参数集合
            BestScore_t     每个线程中的最优个体得分
            BestScore_t_all 所有线程中的最优个体得分集合

            BESTSCORE       所有线程中的最优个体得分
            BESTParam       所有线程中的最优个体参数

            reward_father  所有线程的父代适应度集合
            reward_child   所有线程的子代适应度集合
        '''
        super().__init__(**kwargs)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.cpus = self.comm.Get_size()
        # 超参数设置
        self.args = kwargs
        self.lam = self.cpus - 1
        self.algoname = 'NCSC'
        # self.k = k
        self.version = self.args['version'] 
        self.epoch = self.args['epoch']
        self.r = self.args['r']
        self.updateCount = 0

        if self.version in VersionDivideEveryEpoch:
            self.divide_every_iteration = self.epoch
        else:
            self.divide_every_iteration = 1

        # seed is defined at base class
        # self.randomSeed = np.random.randint(100000)
        # self.rs = np.random.RandomState(self.randomSeed)
        if self.rank == 0:
            self.group_decomposer = Decomposer([2,3,4],self.randomSeed)
        self.logger = Logger(self.logPath())
        self.checkpoint_interval = self.args["checkpoint_interval"]       

        # 同步不同线程的参数到param_all变量中
        self.param = self.policy.get_parameters()
        self.n = len(self.param)
        self.param_all = np.empty((self.cpus, self.n))
        self.comm.Allgather([self.param, MPI.DOUBLE], [self.param_all, MPI.DOUBLE])

        self.param_new = np.zeros(self.n)
        self.sigma = np.ones(self.n) * self.sigma0
        self.sigma_all = np.ones((self.cpus, self.n))
        self.BestParam_t = self.param.copy()
        self.BestParam_t_all = self.param_all.copy()
        self.BestScore_t = np.zeros(1)
        self.BestScore_t_all = np.zeros((self.cpus, 1))
        # BEST in all threads
        self.BESTSCORE = 0
        self.BESTSCORE_id = 0
        self.BESTParam = np.empty(self.n)

        # random seed pool(overwrite None in base class)
        self.train_set, self.test_set = self.getSeedPool(self.args['n_train'],self.args['n_test'],self.randomSeed)

<<<<<<< HEAD
        
=======
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        self.logBasic()
        self.firstEvaluation()
        self.reward_child = None 

        self.log_retest = {
            'steps':[],
            'performance':[]
        }

    def getSeedPool(self,n_train,n_test,seed=None,zero_shot=True,range=1e5):
        """Create train and test random seed pool.  
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

    def firstEvaluation(self):
        """初始化种群后对个体进行评估
        """
        vb = self.initVbn()
        self.logger.save_vb(vb)
        msg = self.rollout(self.param)
        results = np.empty((self.cpus, 2))
        self.comm.Allgather([msg, MPI.DOUBLE], [results, MPI.DOUBLE])
        self.reward_father = results[:,:1].flatten()
        self.BestScore_t[0] = msg[0]
        self.comm.Allgather([self.BestScore_t, MPI.DOUBLE], [self.BestScore_t_all, MPI.DOUBLE])
<<<<<<< HEAD
        self.udpateBEST()
=======
        self.updateBEST()
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

    def calLlambda(self, steps_passed):
        return super().calLlambda(steps_passed)

    def run(self):
        """算法类主循环
        """
        self.start_time = time.time()
        steps_passed = 0
        self.iteration = 1

        # for retestBestFound
        self.last_retest_steps = 0

        while steps_passed <= self.steps_max:
            iter_start_time = time.time()
            steps_this_iter = 0
            self.calLlambda(steps_passed)
            self.divideParameters()

            if (self.iteration -1) % self.divide_every_iteration == 0:
                # 每epoch代的第一代需要对更新次数统计变量self.updateCount进行清0
                self.updateCount = np.zeros(self.n, dtype=np.int32)

            for group_id in range(self.group_number):
                reward_child , cost_steps = self.evalChild(group_id) 
                
                self.logger.log_for_debug(str(self.calDist(self.param))+":"+str(self.rank))
                self.reward_child = reward_child
            
                self.updateBEST()
                self.updateFather()

                steps_this_iter += cost_steps
                steps_passed += cost_steps

            self.updateSigma()
            self.log(iter_start_time, steps_this_iter, steps_passed)
            self.iteration += 1
<<<<<<< HEAD
            self.sampleBest(steps_passed, interval=490000)
=======
            if self.args['env_type'] == 'function':
                self.sampleBest(steps_passed, interval=100000)
            else:
                self.sampleBest(steps_passed, interval=490000)
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

    def divideParameters(self):
        """参数分组

        注意：只有主线程(rank=0)进行参数分组，其余线程只需要同步分组方案和数目即可

        更新了类的参数：
            group_mask     分组方案的保存变量，为长度为n的向量，
                           取值为[0,1,2,3...m-1]，m为分组的数目
            group_number   分组的数目
        """
        # 首先判断是否是每epoch代的第一代
        if (self.iteration - 1) % self.divide_every_iteration == 0: 
            if self.rank == 0:
                group_number, self.group_mask = self.group_decomposer.get_mask(self.n)
                group_number_dict = {"m": group_number}
            else:
                self.group_mask = np.zeros(self.n)
                group_number_dict = None
            self.comm.Bcast([self.group_mask, MPI.DOUBLE], root=0)
            # Broadcast data from one member to all members of a group.
            group_number_dict = self.comm.bcast(group_number_dict, root=0)
            self.group_number = group_number_dict['m']

    def genChild(self,group_id):
        """rewrite"""
        if self.version in VersionUsingBestfound:
            param_newpart1 = self.param.copy() + self.rs.randn(self.n) * self.sigma
            self.param_new = param_newpart1 * (self.group_mask==group_id) + self.BestParam_t * (self.group_mask!=group_id)
        else:
            self.param_new = self.param.copy() + self.rs.randn(self.n) * self.sigma * (self.group_mask == group_id)

    def evalChild(self, group_id):
        """rewrite"""
        cost_steps = 0
        if self.rank != 0:
            self.genChild(group_id)

            if self.args['env_type'] == 'function':
                # 处理边界问题
                bound = test_func_bound[self.args['function_id']]
                self.param_new[self.param_new<bound[0]] = bound[0]
                self.param_new[self.param_new>bound[1]] = bound[1]
           
           # 评估子代
            msg_new = self.rollout(self.param_new,test=False)
            reward_child_t = msg_new[0]
            steps_cost_child = msg_new[1]
            self.updateBest_t(msg_new[0], self.param_new)

            if self.version in VersionUsingBestfound:
                # evaluate tmpx for father 
                tmp_param = self.param.copy() * (self.group_mask==group_id) + self.BestParam_t * (self.group_mask!=group_id)
                tmp_msg = self.rollout(tmp_param,test=False)
                self.updateBest_t(tmp_msg[0], tmp_param)
                reward_father_t = tmp_msg[0]
                steps_cost_father = tmp_msg[1]
        else:
            # Empty array, evaluation results are not used for the update
            reward_child_t, reward_father_t = 0, 0
            steps_cost_father, steps_cost_child = 0, 0

        tmp_steps_cost = None
        # sync child reward
        reward_child = self.syncOneValue(reward_child_t)
        
        if self.version in VersionUsingBestfound:
            # sync father reward
            self.reward_father = self.syncOneValue(reward_father_t)
            tmp_steps_cost = self.syncOneValue(steps_cost_father)

        tmp_steps_cost = self.syncOneValue(steps_cost_child)
        cost_steps += np.sum(tmp_steps_cost)
        return reward_child,cost_steps
        
    def sampleBest(self,steps_passed, interval):
        """在base基础上增加保存网络"""
        super().sampleBest(steps_passed, interval)
        if steps_passed - self.last_retest_steps > interval:
            if self.rank == 0:
                self.logger.save_parameters(self.BESTParam, self.iteration)

    def testBest(self):
        """在base类上加了个log"""
        super().testBest()
        if self.rank == 0:
            logger = self.logger
            logger.log('Final'.ljust(25) + '%e' % self.final_eval)
            logger.save_parameters(self.BESTParam, self.iteration)
            time_elapsed = (time.time() - self.start_time)/60
            logger.log('TimeSinceStart'.ljust(25) + '%f' % time_elapsed)  

    def updateFather(self):
        """没有重写"""
        return super().updateFather()

    def updateSigma(self):
        """1/5法则，没有重写"""
        return super().updateSigma()

    # log
    def log(self, iter_start_time, steps_this_iter, steps_passed):
        """日志函数
        """
        logger = self.logger
        if self.rank == 0:
            if self.iteration % 1000 == 0:
                logger.log('bestscore for every population:%s' % str(self.BestScore_t_all.flatten()))
                logger.log("the best of iteration %d are %e" %(self.iteration, self.BESTSCORE))
            if self.args['env_type'] == 'atari':
                iteration_time = (time.time() - iter_start_time)
                time_elapsed = (time.time() - self.start_time)/60
                # train_mean_rew = np.mean(rews)
                # train_max_rew = np.max(rews)
                logger.log('------------------------------------')
                logger.log('Iteration'.ljust(25) + '%d' % self.iteration)
                logger.log('bestscore for every population:%s' % str(np.around(self.BestScore_t_all.flatten(),decimals=2)))
                logger.log('StepsThisIter'.ljust(25) + '%d' % steps_this_iter)
                logger.log('StepsSinceStart'.ljust(25)+'%d' %steps_passed)
                logger.log('IterationTime'.ljust(25) + '%d' % iteration_time)
                logger.log('TimeSinceStart'.ljust(25) + '%d' %time_elapsed)
                logger.log('Best'.ljust(25) + '%.2f' % self.BESTSCORE)
            if self.iteration % self.checkpoint_interval == 0:
                logger.save_parameters(self.BESTParam, self.iteration)
                self.saveLog('log_retest',self.log_retest)
                self.saveLog('log_train',self.log_train)

    def logPath(self):
        """返回日志的路径，重写
        """
        if self.args['env_type'] == 'atari':
            return "logs_mpi/%s/%s/lam%d/%s" %(self.args['game'], self.algoname, self.lam, self.args['run_name'])
        elif self.args['env_type'] == 'function':
            return "logs_mpi/function%s/%s/lam%d/%s" %(self.args['func_id'], self.algoname, self.lam, self.args['run_name'])

    def logBasic(self):
        """基础信息的日志输出
        """
        logger = self.logger
        if self.rank == 0:
            logger.log("N:%d" % self.lam)
            logger.log("k: %d" % self.k)
            logger.log("sigma0: %s" % str(self.sigma0))
            logger.log("epoch: %d" % self.epoch)
            logger.log("r:%f" % self.r)
            logger.log("stepMax:%d" % self.steps_max)
            logger.log("interval:%d" % self.checkpoint_interval)
            logger.log("randomseed:%d" % self.randomSeed)
            logger.log("trainset size:%d" % self.args['n_train'])
            logger.log("testset size:%d" % self.args['n_test'])

    def log_traincurve(self):
        """save train curve png"""
        self.logger.draw_single(
            self.log_retest,
            self.algoname,
            self.args["game"],
<<<<<<< HEAD
            "train-curve"
=======
            "test-curve"
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        )
        self.logger.draw_single(
            self.log_train,
            self.algoname,
            self.args["game"],
<<<<<<< HEAD
            "test-curve"
=======
            "train-curve"
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        )
        self.logger.draw_two(
            self.log_retest,
            self.log_train,
            self.algoname,
            self.args["game"],
            "train-test-curve"
        )

'''
NCS算法的入口文件，用于处理输入参数。
输入参数：
    --run_name, -r    本次运行的别名，保存日志文件时会用到这个名字
    --env_type, -e    环境的类型（取值为 atari或者function）
    --game,-g         Atari游戏的名字，不需要带NoFrameskip-v4，代码中会自动加具体的版本信息
    --func_id, -f CEC测试函数的id，取值范围[1,2,3,4,5,6]
    --func_dim, -d   CEC测试函数的维度
    --k, -k           演化过程中得到模型适应度值时模型被评估的次数
    --epoch,          NCS算法中更新高斯噪声标准差周期，一般取值为5的倍数
    --sigma0,         NCS算法中高斯噪声标准差的初始值
    --rvalue          NCS算法中更新高斯噪声标准差的系数
'''

@click.command()
@click.option('--run_name', '-r', required=True, default = 'debug',type=click.STRING, help='Name of the run, used to create log folder name')
@click.option('--env_type', '-e', required=True, default = 'atari',type=click.Choice(['atari', 'function']))
@click.option('--game', '-g', type=click.STRING, default = 'Alien', help='game name for RL')
@click.option('--func_id', '-f', type=click.INT, help='function id for function optimization')
@click.option('--func_dim', '-d', type=click.INT, help='the dimension of solution for function optimization')
@click.option('--config','-c', default='./config/NCSC-1/atari-opt.json', help='configuration file path')
def main(run_name, env_type, game, func_id, func_dim, config):
    with open(config, 'r') as f:
        kwargs = json.loads(f.read())
    kwargs['run_name'] = run_name
    kwargs['env_type'] = env_type
    kwargs['game'] = game
    kwargs['func_id'] = func_id
    kwargs['func_dim'] = func_dim

    algo = NCSCCAlgo(**kwargs)
    algo.run()
    # 最优解的最终测试以及保存日志
    algo.testBest()
    algo.log_traincurve()

  

if __name__ == '__main__':
    main()


