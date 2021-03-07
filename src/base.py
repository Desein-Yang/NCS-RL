
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base.py
@Time    :   2020/06/27 16:04:57
@Describtion: 演化强化学习算法基类
'''

import time
import gym
import pickle
import click
import os
import numpy as np
import matplotlib.pyplot as plt 
from mpi4py import MPI
from src.env_wrappers import wrap_dqn
from src.policy import Policy,FuncPolicy
from src.logger import Logger
from src.testfunc import test_func_bound


class BaseAlgo(object):
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
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.cpus = self.comm.Get_size()

        # 超参数设置
        self.args = kwargs
        self.k = self.args['k']
        self.lr = self.args['learning_rate']
        self.sigma0 = self.args['sigma0']
        self.algoname = 'None'

        if self.args['seed'] == 0:
            self.randomSeed = np.random.randint(100000)
        else:
            self.randomSeed = self.args['seed']
        self.rs = np.random.RandomState(self.randomSeed)
        # self.logger = Logger(self.logPath())
        
        # 创建策略模型以及设置对应的超参数
        if self.args['env_type'] == 'atari':
            env = gym.make("%sNoFrameskip-v4" % self.args['game'])
            human = 1 if self.args["human_start"] == True else 0
            env = wrap_dqn(env,human_start=human,frame=self.args["frame_max"])
            self.policy = Policy(
                env,
                network=self.args['network'], 
                nonlin_name=self.args['nonlin_name']
            )
            self.steps_max = self.args['steps_max']
        elif self.args['env_type'] == 'function':
            self.policy = FuncPolicy(
                self.args['func_dim'], 
                self.args['func_id'], 
                self.rank, 
                self.randomSeed)
            self.steps_max = 5000 * self.args['func_dim']
            bound = test_func_bound[self.args['func_id']]
            self.sigma0 = (bound[1] - bound[0]) / (self.cpus - 1)

        # 同步不同线程的参数到param_all变量中
        self.param = self.policy.get_parameters()
        self.n = len(self.param)
        self.param_all = np.empty((self.cpus, self.n))
        self.comm.Allgather(
            [self.param, MPI.DOUBLE], 
            [self.param_all, MPI.DOUBLE]
        )

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
        self.BESTParam = np.zeros(self.n)

        # self.logBasic()
        # self.firstEvaluation()
        self.train_set = None
        self.test_set = None
        self.reward_child = None 

        self.steps_passed = 0
        self.last_retest_steps = 0
        self.log_retest = {
            'steps':[],
            'performance':[]
        }
        self.log_train = {
            'steps':[],
            'performance':[]
        }
        

    # main loop
    def run(self):
        """特定算法类的运行函数
        """
        raise NotImplementedError

    # NCS framework
    def initVbn(self):
        """VBN的初始化，第一次执行policy时使用"""
        if self.args['env_type']=='atari':
            vb = self.policy.get_vb()
            self.comm.Bcast([vb, MPI.FLOAT], root=0)
            self.policy.set_vb(vb)
            return vb
    
    def firstEvaluation(self):
        """初始化种群后对个体进行评估初始化
        """
        raise NotImplementedError

    def genChild(self):
        """产生子代的规则，默认为高斯分布产生
        """
        self.param_new = self.param + self.rs.normal(scale = self.sigma,size = self.n)

    def evalChild(self):
        """评估子代

        非主线程都要产生一个子代，并进行评估，完成之
        后同步子代的适应度和消耗的训练帧数

        返回值：
            cost_steps ： 消耗的训练帧数
            self.reward_child : 更新后的子代reward序列
        """
        cost_steps = 0
        if self.rank != 0:
            # Generate child
            self.genChild()

            # Evaluate child
            msg_new = self.rollout(self.param_new,test=False)
            reward_child_t = msg_new[0]
            steps_cost_child = msg_new[1]
            self.updateBest_t(msg_new[0], self.param_new)
        else:
            # Empty array, evaluation results are not used for the update
            reward_child_t, reward_father_t = 0, 0
            steps_cost_father, steps_cost_child = 0, 0

        tmp_steps_cost = None
        # sync child reward
        reward_child = self.syncOneValue(reward_child_t)
        tmp_steps_cost = self.syncOneValue(steps_cost_child)
        cost_steps += np.sum(tmp_steps_cost)
        return reward_child, cost_steps

    def rollout(self, parameters, test=False):
        """对策略中的rollout的封装，支持k次评估 & Train/test split.  
        返回值：
            msg: [mean_reward, sum_len]
                第一个为平均得分
                第二个为消耗的训练帧总和
        """
        assert self.train_set is not None, "train set are empty"
        assert self.test_set is not None, "test set are empty"
        lens = [0]
        rews = [0] 
        e_r = 0
        e_l = 0
        self.policy.set_parameters(parameters)
        for j in range(self.k):
            if test:
                env_seed = np.random.choice(self.test_set)
            else:
                env_seed = np.random.choice(self.train_set)
            e_rew, e_len = self.policy.rollout(seed=env_seed)
            e_r += e_rew
            e_l += e_len
        rews[0] = e_r/self.k
        lens[0] = e_l
        msg = np.array(rews + lens)
        return msg

    def updateBEST(self):
        """依据每个线程中的最优个体来得到所有线程中的最优个体
        """
        self.comm.Allgather([self.BestParam_t, MPI.DOUBLE], [self.BestParam_t_all, MPI.DOUBLE])
        self.comm.Allgather([self.BestScore_t, MPI.DOUBLE], [self.BestScore_t_all, MPI.DOUBLE])
        self.BESTSCORE = np.max(self.BestScore_t_all.flatten()[1:])
        self.BESTSCORE_id = np.argmax(self.BestScore_t_all.flatten()[1:])+1
        self.log_train['steps'].append(self.steps_passed)
        self.log_train['performance'].append(self.BESTSCORE)
        self.BESTParam = self.BestParam_t_all[self.BESTSCORE_id].copy()

    def updateFather(self):
        """如何替换父代
        """
        return NotImplementedError

    def updateLr(self):
        """更新学习率，可选"""
        return NotImplementedError

    def updateBest_t(self, score, param):
        """更新当前线程的最优个体得分与参数
        """
        if score > self.BestScore_t[0]:
            self.BestScore_t[0] = score
            self.BestParam_t = param.copy()

    def sampleBest(self, interval):
        """以一定interval重新测试保存的所有线程的最优解，画演化曲线 self.cpus * k 次
        """
        if self.steps_passed - self.last_retest_steps > interval:
            self.last_retest_steps = self.steps_passed
            msg_new = self.rollout(self.BESTParam,test=True)
            
            reward = msg_new[0]
            reward_all = self.syncOneValue(reward)
            reward_mean = np.mean(reward_all)
            self.log_retest['steps'].append(self.steps_passed)
            self.log_retest['performance'].append(reward_mean)

    def testBest(self, repeat = 200):
        """运行算法之后，对最好的个体进行测试200次，并保存结果
        """
        # logger = self.logger
        # print("save And Test best")
        # test best
        self.k = 1
        # 计算每个线程跑多少次游戏
        test_times = int(repeat / self.cpus)
        # 剩余的给主线程来跑
        reminder = repeat - test_times * self.cpus
        final_rews = []
        for i in range(test_times):
            msg = self.rollout(self.BESTParam,test=True)
            final_rews.append(msg[0])
        part_one_reward = self.syncOneValue(np.mean(final_rews))

        if self.rank == 0:
            part_two_reward = []
            for i in range(reminder):
                msg = self.rollout(self.BESTParam, test=True)
                part_two_reward.append(msg[0])
            self.final_eval = (np.sum(part_two_reward) + np.sum(part_one_reward) * test_times) / repeat
            self.saveLog('log_retest',self.log_retest)
            self.saveLog('log_train',self.log_train)
    # log util
    def logPath(self):
        """返回日志的路径
        """
        return NotImplementedError

    def log(self):
        """日志函数，需要被重写
        """
        raise NotImplementedError

    def logBasic(self):
        """基础信息的日志输出，需要被重写
        """
        raise NotImplementedError

    def calDist(self,params):
        """ 输出参数的均值、最大最小值、方差等统计信息观察"""
        mean_ = np.around(np.mean(params),decimals=2)
        min_ = np.around(np.min(params),decimals=2)
        max_ = np.around(np.max(params),decimals=2)
        sigma = np.around(np.var(params),decimals=2)
        return mean_,min_,max_,sigma

    def saveLog(self,filename,log):
        """保存重新测试的日志
        """
        filepath = os.path.join(self.logPath(), filename + '.pickle')
        with open(filepath, 'wb') as f:
            pickle.dump(log, f)

    # mpi utils 
    def syncOneValue(self, v):
        """工具函数，用于同步每个线程的单个标量值到每个线程的向量中。
        
        对mpi的Allgather的简单封装

        参数：
            v：标量
        返回值：
            np.ndarray ： 大小为 cpus
        """
        v_t = np.array([v])
        v_all = np.zeros((self.cpus, 1))
        self.comm.Allgather([v_t, MPI.DOUBLE], [v_all, MPI.DOUBLE])
        return v_all.flatten()

    def syncOneVector(self, v):
        """工具函数，用于同步每个线程的矢量值到同名总的矢量中
        
        对mpi的简单封装
        
        """
        v_t = np.array(v,dtype=np.float)
        v_all = np.zeros((self.cpus, v_t.shape[0]))
        self.comm.Allgather([v_t, MPI.DOUBLE], [v_all, MPI.DOUBLE])
        return v_all

:


class NCSAlgo(BaseAlgo):
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
        self.k = self.args['k']
        self.epoch = self.args['epoch']
        self.sigma0 = self.args['sigma0']
        self.r = self.args['r']
        self.updateCount = 0
        self.algoname = 'NCSFrame'

    # main loop
    def run(self):
        """特定算法类的运行函数
        """
        raise NotImplementedError

    # NCS framework
    def initVbn(self):
        """VBN的初始化，第一次执行policy时使用"""
        return super().initVbn()
    
    def firstEvaluation(self):
        """初始化种群后对个体进行评估初始化
        """
        msg = self.rollout(self.param,test=False)
        results = np.empty((self.cpus, 2))
        self.comm.Allgather([msg, MPI.DOUBLE], [results, MPI.DOUBLE])
        self.reward_father = results[:,:1].flatten()
        self.BestScore_t[0] = msg[0]
        self.comm.Allgather([self.BestScore_t, MPI.DOUBLE], [self.BestScore_t_all, MPI.DOUBLE])
        self.updateBEST()

    def genChild(self):
        """产生子代的规则，默认为高斯分布产生
        """
        self.param_new = self.param + np.random.normal(scale = self.sigma,size = self.n)

    def evalChild(self):
        """评估子代

        非主线程都要产生一个子代，并进行评估，完成之
        后同步子代的适应度和消耗的训练帧数

        返回值：
            cost_steps ： 消耗的训练帧数
            self.reward_child : 更新后的子代reward序列
        """
        cost_steps = 0
        if self.rank != 0:
            # Generate child
            self.genChild()

            # Evaluate child
            msg_new = self.rollout(self.param_new,test=False)
            reward_child_t = msg_new[0]
            steps_cost_child = msg_new[1]
            self.updateBest_t(msg_new[0], self.param_new)
        else:
            # Empty array, evaluation results are not used for the update
            reward_child_t, reward_father_t = 0, 0
            steps_cost_father, steps_cost_child = 0, 0

        tmp_steps_cost = None
        # sync child reward
        reward_child = self.syncOneValue(reward_child_t)
        tmp_steps_cost = self.syncOneValue(steps_cost_child)
        cost_steps += np.sum(tmp_steps_cost)
        return reward_child, cost_steps

    # def rollout(self, parameters):

    def updateFather(self):
        """ 更新父代

        根据算法计算是否用子代替换父代，只有非主线程参与  
        需要做的事情：  
            1. 适应度归一化  
            2. 计算相关性，并归一化  
            3. 是否替换  
            4. 是否需要更新父代个体的适应度值  
            5. 同步父代的参数  
        """
        if self.rank != 0:
            # refer to NCSCC pseudo code line 12
            father_corr = self.calCorr(self.param_all, self.param, self.sigma_all, self.sigma)
            child_corr = self.calCorr(self.param_all, self.param_new, self.sigma_all, self.sigma)

            # 每个线程计算自己的correlation和new correlation， 但是对于相关性和fitness都需要进行归一化
            child_corr = child_corr / (father_corr + child_corr)
            # 优化目标是到最小
            father_f = self.reward_father[self.rank] - self.BESTSCORE + 10**-10
            child_f = self.reward_child[self.rank] - self.BESTSCORE + 10**-10
            child_f = child_f / (child_f + father_f)
            
            if child_f / child_corr < self.llambda:
            
                # 抛弃旧的解，更换为新解
                self.param = self.param_new.copy()
                self.updateCount = self.updateCount + 1
                self.reward_father[self.rank] = self.reward_child[self.rank]
            reward_father_t = self.reward_father[self.rank]
        else:
            reward_father_t = 0
        self.param_all = self.syncOneVector(self.param)

    def updateSigma(self):
        """更新协方差，并同步 
        """
        if self.rank != 0:
            if self.iteration % self.epoch == 0:
                self.sigma[self.updateCount/self.epoch<0.2] = self.sigma[self.updateCount/self.epoch<0.2] * self.r
                self.sigma[self.updateCount/self.epoch>0.2] = self.sigma[self.updateCount/self.epoch>0.2] / self.r
        # if self.iteration % self.epoch == 0:
        self.comm.Allgather([self.sigma, MPI.DOUBLE], [self.sigma_all, MPI.DOUBLE])

    # def updateBEST(self):
    # def updateBest_t(self, score, param):
    # def sampleBest(self, steps_passed, interval):

    # log util
    def logPath(self):
        """return log path
        """
        if self.args['env_type'] == 'atari':
            return "logs_mpi/%s/%s/lam%d/%s" %(self.algoname,self.args['game'], self.cpus, self.args['run_name'])
        elif self.args['env_type'] == 'function':
            return "logs_mpi/%s/function%s/lam%d/%s" %(self.algoname,self.args['func_id'], self.cpus, self.args['run_name'])

    def log(self):
        """日志函数，需要被重写
        """
        raise NotImplementedError

    def logBasic(self):
        """基础信息的日志输出，需要被重写
        """
        logger = self.logger
        if self.rank == 0:
            logger.log("N:%d" % self.lam)
            logger.log("k: %d" % self.k)
            logger.log("sigma0: %s" % str(self.sigma0))
            logger.log("epoch: %d" % self.epoch)
            logger.log("r:%f" % self.r)
            logger.log("stepMax:%d" % self.steps_max)

    # def saveLog(self,filename,log):

    # math utiils
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
        small_value = 1e-8
        part1 = 1 / 8 * np.sum(xi_xj * xi_xj / (big_sigma + small_value))
        part2 = (
            np.sum(np.log(big_sigma + small_value))
            - 1 / 2 * np.sum(np.log(big_sigma1 + small_value))
            - 1 / 2 * np.sum(np.log(big_sigma2 + small_value))
        )
        return part1 + 1 / 2 * part2

    def calCorr(self, params_list, param, sigma_all, sigma):
        """计算分布param的相关性
        参数：
            n(int): the number of parameters
            
            param(np.ndarray): 当前分布的均值
            sigma(np.ndarray): 当前分布的协方差

            param_list(np.ndarray): 所有分布的均值
            sigma_all(np.ndarray): 所有分布的协方差

            rank(int): 当前线程的id
        返回值：
            这个分布的相关性
        """
        DBlist = []
        for i in range(len(params_list)):
            # i 是该进程在所有进程中序号
            if i != self.rank:
                param2 = params_list[i]
                sigma2 = sigma_all[i]
                DB = self.calBdistance(param, param2, sigma, sigma2)
                DBlist.append(DB)
        return np.min(DBlist)

    def calLlambda(self):
        """计算 llambda的值，这里采用llambda表示算法中的lambda，因为lambda在python中是一个关键字
        """
        self.llambda = self.rs.rand() * (0.1-0.1*self.steps_passed/self.steps_max) + 1.0

    # mpi utils 
    # def syncOneValue(self, v):
    # def syncOneVector(self, v)
