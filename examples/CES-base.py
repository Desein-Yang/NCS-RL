#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CES-base.py
@Time    :   2020/06/27 16:41:31
@Author  :   Qi Yang
@Describtion:  CES 算法 (调用base类的)
'''

# here put the import lib
import time
import gym
import pickle
import click
import os
import numpy as np
from mpi4py import MPI
from src.policy import Policy,FuncPolicy
from src.logger import Logger
from src.testfunc import test_func_bound
from src.base import BaseAlgo

class CESAlgo(BaseAlgo):
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
        
        # Mpi
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.cpus = self.comm.Get_size()

        # 超参数设置
        self.args = kwargs
        self.lam = self.cpus - 1
        self.algoname = 'CES'
        self.k = self.args['k']
        # self.epoch = self.args['epoch']
        self.sigma0 = 1.0
        # self.r = self.args['r']
        # self.updateCount = 0
        self.lr_decay = True
        self.lr_decay_rate = 0.99

        self.randomSeed = np.random.randint(100000)
        self.rs = np.random.RandomState(self.randomSeed)
        self.logger = Logger(self.logPath())
        self.checkpoint_interval = self.args['checkpoint_interval']
        
        # 创建策略模型以及设置对应的超参数
        # if self.args['env_type'] == 'atari':
        #     env = gym.make("%sNoFrameskip-v4" % self.args['game'])
        #     self.policy = Policy(env, network=self.args['network'], nonlin_name=self.args['nonlin_name'])
        #     vb = self.policy.get_vb()
        #     self.comm.Bcast([vb, MPI.FLOAT], root=0)
        #     self.policy.set_vb(vb)
        #     self.logger.save_vb(vb)
        #     self.steps_max = 25*10**6
        # elif self.args['env_type'] == 'function':
        #     self.policy = FuncPolicy(self.args['D'], self.args['function_id'], self.rank, self.randomSeed)
        #     self.steps_max = 5000 * self.args['D']
        #     bound = test_func_bound[self.args['function_id']]
        #     self.sigma0 = (bound[1] - bound[0]) / self.lam

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

        self.logBasic()
        self.firstEvaluation()
        self.reward_child = None 

        self.log_retest = {
            'steps':[],
            'performance':[]
        }

    def firstEvaluation(self):
        """初始化种群后对个体进行评估
        """
        self.initVbn()
        msg = self.rollout(self.param)
        results = np.empty((self.cpus, 2))
        self.comm.Allgather([msg, MPI.DOUBLE], [results, MPI.DOUBLE])
        self.reward_father = results[:,:1].flatten()
        self.BestScore_t[0] = msg[0]
        self.comm.Allgather([self.BestScore_t, MPI.DOUBLE], [self.BestScore_t_all, MPI.DOUBLE])
        self.udpateBEST()

    def run(self):
        """算法类主循环重写
        1. Generate lam individuals with Gaussian noise (0,I)
        2. Evaluate by rollout() in RL
        3. Gather results from all processes
        4. Sort and Select 
        5. Update Best
        """
        self.start_time = time.time()
        steps_passed = 0
        self.iteration = 1

        # for retestBestFound
        self.last_retest_steps = 0

        while steps_passed <= self.steps_max:
            iter_start_time = time.time()
            steps_this_iter = 0

            # generate child and evaluate it
            reward_child, cost_steps = self.evalChild()
            self.reward_child = reward_child

            self.udpateBEST()
            self.updateFather()
            steps_this_iter += cost_steps
            steps_passed += cost_steps

            self.log(iter_start_time, steps_this_iter, steps_passed)
            self.iteration += 1
            self.sampleBest(steps_passed, interval=490000)

    def genChild(self):
        """产生子代的规则，默认为高斯分布产生  
        CES 重写为（0，I）
        """
        self.param_new = self.param + self.rs.normal(scale = self.sigma,size = self.n)

    def evalChild(self):
        """没有重写"""
        return super().evalChild()

    def sampleBest(self, steps_passed, interval):
        """在base基础上保存网络
        """
        super().sampleBest(steps_passed, interval)
        if steps_passed - self.last_retest_steps > interval:
            if self.rank == 0:
                self.logger.save_parameters(self.BESTParam, self.iteration)


    def udpateBEST(self):
        """依据每个线程中的最优个体来得到所有线程中的最优个体
        """
        self.comm.Allgather([self.BestParam_t, MPI.DOUBLE], [self.BestParam_t_all, MPI.DOUBLE])
        self.comm.Allgather([self.BestScore_t, MPI.DOUBLE], [self.BestScore_t_all, MPI.DOUBLE])
        self.BESTSCORE = np.max(self.BestScore_t_all.flatten()[1:])
        self.BESTSCORE_id = np.argmax(self.BestScore_t_all.flatten()[1:])+1
        self.BESTParam = self.BestParam_t_all[self.BESTSCORE_id].copy()        
    
    def updateFather(self):
        """ CES 根据一个公式对每一个赋权值加和更新  
        修改父代参数 self.param
        """
        if self.rank != 0:
            w = self.calweight()
            for i in range(self.lam):
                self.param = self.param + self.lr * w[i] * (self.param_new - self.param)
                self.updateLr()
        else:
            self.param = self.param
        self.param_all = self.syncOneVector(self.param)
    
    def updateLr(self):
        """学习率衰减 0.99"""
        if self.lr_decay is True:
            self.lr = self.lr_decay_rate * self.lr

    # weight calculate
    def calweight(self):
        rank = np.argsort(self.reward_child[1:]) + 1 #返回每个子代的编号,rank0 不包含在内
        tmp = [np.log(self.lam+0.5)-np.log(j) for j in rank]
        weight = tmp / np.sum(tmp)
        return weight

    def testBest(self):
        """在base类上加了个log"""
        super().testBest()
        if self.rank == 0:
            logger = self.logger
            logger.log('Final'.ljust(25) + '%e' % self.final_eval)
            logger.save_parameters(self.BESTParam, self.iteration)
            time_elapsed = (time.time() - self.start_time)/60
            logger.log('TimeSinceStart'.ljust(25) + '%f' % time_elapsed) 
            
    # log
    def log(self, iter_start_time, steps_this_iter, steps_passed):
        """日志函数
        """
        logger = self.logger
        if self.rank == 0:
            if self.iteration % 1000 == 0:
                logger.log('bestscore for every population:%s' % str(np.around(self.BestScore_t_all.flatten(),decimals=2)))
                logger.log("the best of iteration %d are %e" %(self.iteration, self.BESTSCORE))
            if self.args['env_type'] == 'atari':
                iteration_time = (time.time() - iter_start_time)
                time_elapsed = (time.time() - self.start_time)/60
                # train_mean_rew = np.mean(rews)
                # train_max_rew = np.max(rews)
                logger.log('------------------------------------')
                logger.log('Iteration'.ljust(25) + '%d' % self.iteration)
                logger.log('bestscore for every population:%s' % str(np.around(self.BestScore_t_all.flatten(),decimals=2)))
                logger.log('StepsThisIter'.ljust(25) + '%f' % steps_this_iter)
                logger.log('StepsSinceStart'.ljust(25)+'%f' %steps_passed)
                logger.log('IterationTime'.ljust(25) + '%f' % iteration_time)
                logger.log('TimeSinceStart'.ljust(25) + '%d' %time_elapsed)
                logger.log('Best'.ljust(25) + '%f' % self.BESTSCORE)
            if self.iteration % self.checkpoint_interval == 0:
                logger.save_parameters(self.BESTParam, self.iteration)
                self.saveLog('log_retest',self.log_retest)

    def logPath(self):
        """返回日志的路径，重写
        """
        if self.args['env_type'] == 'atari':
            return "logs_mpi/%s/%s/lam%d/%s" %(self.args['game'], self.algoname, self.lam, self.args['run_name'])
        elif self.args['env_type'] == 'function':
            return "logs_mpi/function%s/%s/lam%d/%s" %(self.args['function_id'], self.algoname, self.lam, self.args['run_name'])

    def logBasic(self):
        """基础信息的日志输出
        """
        logger = self.logger
        if self.rank == 0:
            logger.log("N:%d" % self.lam)
            logger.log("k: %d" % self.k)
            logger.log("sigma0: %s" % str(self.sigma0))
            logger.log("stepMax:%d" % self.steps_max)
            logger.log("random seed: %d" % self.randomSeed)


'''
CES算法的入口文件，用于处理输入参数。
输入参数：
    --run_name, -r    本次运行的别名，保存日志文件时会用到这个名字
    --env_type, -e    环境的类型（取值为 atari或者function）
    --game,-g         Atari游戏的名字，不需要带NoFrameskip-v4，代码中会自动加具体的版本信息
    --function_id, -f CEC测试函数的id，取值范围[1,2,3,4,5,6]
    --dimension, -d   CEC测试函数的维度
    --k, -k           演化过程中得到模型适应度值时模型被评估的次数
    --step_max, -s    演化总帧数限制
    --lr              学习率初始值
'''

@click.command()
@click.option('--run_name', '-r', required=True, default = 'debug',type=click.STRING, help='Name of the run, used to create log folder name')
@click.option('--env_type', '-e', default = 'atari',type=click.Choice(['atari', 'function']))
@click.option('--game', '-g', type=click.STRING, default = 'Alien', help='game name for RL')
@click.option('--function_id', '-f', type=click.INT, help='function id for function optimization')
@click.option('--dimension', '-d', type=click.INT, help='the dimension of solution for function optimization')
@click.option('--k', '-k', type=click.INT, default=10, help='the number of evaluation times in training')
@click.option('--stepmax','-s', type=click.INT, default=25000000, help='maximum of framecount')
@click.option('--lr', type=click.FLOAT, default=1.0, help='init value of learning rate')
@click.option('--seed', type=click.INT, default=0, help='assign seed(optinal)')
@click.option('--checkpoint_interval', type=click.INT, default=200, help='save chekpoint interval')
def main(run_name, env_type, game, function_id, dimension, k, stepmax,lr,seed,checkpoint_interval):
    # 算法入口
    kwargs = {
        'network': 'Nature',
        'nonlin_name': 'relu',
        'k': k,
        'sigma0': 1,
        'run_name': run_name,
        'env_type': env_type,
        'game': game,
        'function_id': function_id,
        'D': dimension,
        'steps_max':stepmax,
        'learning_rate':lr,
        'seed':seed,
        'checkpoint_interval':checkpoint_interval
    }
    algo = CESAlgo(**kwargs)
    algo.run()
    # 最优解的最终测试以及保存日志
    algo.testBest()
    

if __name__ == '__main__':
    main()



