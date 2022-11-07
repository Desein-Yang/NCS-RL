# /home/yangqi/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   NCS-C-base.py
@Time    :   2020/06/27 16:41:31
@Author  :   Qi Yang
@Describtion:  NCS-C 算法 (调用base类的)
               测试已通过、正常运行
'''

# paper: [1]K. Tang, P. Yang, and X. Yao, “Negatively Correlated Search,” IEEE J. Select. Areas Commun., vol. 34, no. 3, pp. 542–550, Mar. 2016, doi: 10.1109/JSAC.2016.2525458.

# here put the import lib
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



class NCSC2Algo(NCSAlgo):
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
<<<<<<< HEAD
        self.algoname = 'NCSC'
=======
        self.algoname = 'NCSC2'
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
        self.epoch = self.args['epoch']
        self.r = self.args['r']
        self.updateCount = 0
        self.phi=0.0001

        # seed is defined at base class
        # self.randomSeed = np.random.randint(100000)
        # self.rs = np.random.RandomState(self.randomSeed)
        self.logger = Logger(self.logPath())
<<<<<<< HEAD
        self.checkpoint_interval = self.args["checkpoint_interval"]       
=======
        self.checkpoint_interval = self.args["checkpoint_interval"]
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

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
        if self.args["seed_pool"] == True:
            self.train_set, self.test_set = self.getSeedPool(self.args['n_train'],self.args['n_test'],self.randomSeed)
        else:
            self.train_set = np.arange(1e6,dtype=np.int)
            self.test_set = self.train_set

        self.logBasic()
        self.firstEvaluation()



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
        self.updateBEST()

    def calLlambda(self):
        return super().calLlambda()

    def run(self):
        """算法类主循环重写
        """
        self.start_time = time.time()
        self.iteration = 1

        # for retestBestFound
        self.last_retest_steps = 0

        while self.steps_passed <= self.steps_max:
            iter_start_time = time.time()
            steps_this_iter = 0
            self.calLlambda()
            if self.iteration -1 == 0:
                # 每epoch代的第一代需要对更新次数统计变量self.updateCount进行清0
                self.updateCount = np.zeros(self.n, dtype=np.int32)

            # generate child and evaluate it
            reward_child, cost_steps = self.evalChild()
<<<<<<< HEAD
                
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
            self.logger.log_for_debug(str(self.calDist(self.param))+":"+str(self.rank))
            self.reward_child = reward_child

            self.updateBEST()
            self.updateFather()
            steps_this_iter += cost_steps
            self.steps_passed += cost_steps

            self.updateSigma()
            self.log(iter_start_time, steps_this_iter)
            self.iteration += 1
            self.sampleBest(interval=490000)


    def evalChild(self):
        """没有重写"""
        return super().evalChild()
<<<<<<< HEAD
        
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

    def sampleBest(self, interval):
        """在base基础上增加保存网络"""
        super().sampleBest(interval)
        if self.steps_passed - self.last_retest_steps > interval:
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
<<<<<<< HEAD
            logger.log('TimeSinceStart'.ljust(25) + '%f' % time_elapsed)  
=======
            logger.log('TimeSinceStart'.ljust(25) + '%f' % time_elapsed)
            with open('log.csv','w') as f:
                f.write("20200914,"+self.algoname+','+str(self.args["game"])+","+str(self.final_eval)+','+str(time_elapsed))
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

    def updateFather(self):
        """重写
        改为加法"""
        if self.rank != 0:
            # refer to NCSCC pseudo code line 12
            father_corr = self.calCorr(self.param_all, self.param, self.sigma_all, self.sigma)
            child_corr = self.calCorr(self.param_all, self.param_new, self.sigma_all, self.sigma)

            # 每个线程计算自己的correlation和new correlation， 但是对于相关性和fitness都需要进行归一化
            #child_corr = child_corr / (father_corr + child_corr)
            # 优化目标是到最小
            father_f = self.reward_father[self.rank] - self.BESTSCORE + 10**-10
            child_f = self.reward_child[self.rank] - self.BESTSCORE + 10**-10
            #child_f = child_f / (child_f + father_f)
<<<<<<< HEAD
            
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
            if child_f - self.phi* child_corr > father_f - self.phi * father_corr:
                # 抛弃旧的解，更换为新解
                self.param = self.param_new.copy()
                self.updateCount = self.updateCount + 1
                self.reward_father[self.rank] = self.reward_child[self.rank]
            reward_father_t = self.reward_father[self.rank]
        else:
            reward_father_t = 0
        self.param_all = self.syncOneVector(self.param)
<<<<<<< HEAD
)
=======

>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

    def updateSigma(self):
        """1/5法则，没有重写"""
        return super().updateSigma()

    # log
    def log(self, iter_start_time, steps_this_iter):
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
                logger.log('------------------------------------')
                logger.log('Iteration'.ljust(25) + '%d' % self.iteration)
                logger.log('bestscore for every population:%s' % str(np.around(self.BestScore_t_all.flatten(),decimals=2)))
                logger.log('StepsThisIter'.ljust(25) + '%d' % steps_this_iter)
                logger.log('StepsSinceStart'.ljust(25)+'%d' % self.steps_passed)
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
<<<<<<< HEAD
=======
            logger.log("humanstart:%d"% self.args["human_start"])
            logger.log("seedpool%d"% self.args["seed_pool"])
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1

    def log_traincurve(self):
        """save train curve png"""
        self.logger.draw_single(
            self.log_retest,
            self.algoname,
            self.args["game"],
            "train-curve"
        )
        self.logger.draw_single(
            self.log_train,
            self.algoname,
            self.args["game"],
            "test-curve"
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

<<<<<<< HEAD
    algo = NCSCAlgo(**kwargs)
=======
    algo = NCSC2Algo(**kwargs)
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
    algo.run()
    # 最优解的最终测试以及保存日志
    algo.testBest()
    algo.log_traincurve()

<<<<<<< HEAD
  

if __name__ == '__main__':
    main()

=======


if __name__ == '__main__':
    main()
>>>>>>> 3f1f18e2eb8acab8c427897252c0589aeab277c1
