
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   NCNES.py
@Time    :   2020/06/27 16:41:31
@Author  :   Qi Yang
@Describtion:  NCNES 算法类（自定义类不基于base）
'''
import time
import gym
import pickle
import click
import os
import math
import numpy as np

from mpi4py import MPI

from src.policy import Policy,FuncPolicy
from src.logger import Logger
from src.testfunc import test_func_bound
from src.base import BaseAlgo

# 以下三个变量用于描述版本号和具体内容之间的联系
# 如版本1，3 采用Bestfound_i 来填充
VersionUsingBestfound = [1, 3]
VersionUsingFather = [2, 4]
VersionDivideEveryEpoch = [3, 4]

class NCNESAlgo(object):
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
        # self.lam = self.cpus - 1
        self.k = self.args['k'] # eva times
        # self.epoch = self.args['epoch']
        self.sigma0 = self.args['sigma0'] # sigma_init
        self.r = self.args['r']

        # NCNES hyperparams
        self.pop_size = self.args["mu"]
        self.lam = self.args["lam"]
        self.phi = self.args['phi']
        self.lr_sigma = self.args['lr_sigma']
        self.lr_mean = self.args['lr_mean']
        self.lr_decay = True
        self.phi_decay = True
        self.eva_decay = True
        self.parallel = self.args['parallel'] # i,p,s

        # set up random seed 
        # torch.manual_seed(time.time())
        # np.random.seed(111)
        # seed = np.random.randint(0, 1000000)
        self.seed = [np.random.randint(1,1000000) for i in range(self.pop_size)]
        self.logger = Logger(self.logPath())
        self.logger.log("seed:%s" % str(self.seed))
        
        # 创建策略模型以及设置对应的超参数
        if self.args['env_type'] == 'atari':
            env = gym.make("%sNoFrameskip-v4" % self.args['game'])
            self.policy = Policy(env, network=self.args['network'], nonlin_name=self.args['nonlin_name'])
            vb = self.policy.get_vb()
            self.comm.Bcast([vb, MPI.FLOAT], root=0)
            self.policy.set_vb(vb)
            self.logger.save_vb(vb)
            self.steps_max = 25*10**6
        elif self.args['env_type'] == 'function':
            self.policy = FuncPolicy(self.args['D'], self.args['function_id'], self.rank, self.randomSeed)
            self.steps_max = 5000 * self.args['D']
            bound = test_func_bound[self.args['function_id']]
            self.sigma0 = (bound[1] - bound[0]) / self.lam

        # 同步不同线程的参数到param_all变量中
        self.param = self.policy.get_parameters()
        self.n = len(self.param)
        self.param_all = np.empty((self.cpus, self.n))
        # 用这个当作 mean list
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
        self.iteration = 0
        self.eps = 1e-8

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
        msg = self.rollout(self.param)
        results = self.syncOneVector(msg)
        self.reward_father = results[:,:1].flatten()
        self.BestScore_t[0] = msg[0]
        self.udpateBEST()

    def udpateBEST(self):
        """依据每个线程中的最优个体来得到所有线程中的最优个体
        """
        self.comm.Allgather([self.BestParam_t, MPI.DOUBLE], 
                            [self.BestParam_t_all, MPI.DOUBLE])
        self.logger.log(' upBEST_t bestscore for every population:%s' % str(self.BestScore_t))
        
        self.comm.Allgather([self.BestScore_t, MPI.DOUBLE], 
                            [self.BestScore_t_all, MPI.DOUBLE])
        oldBESTSCORE = self.BESTSCORE
        self.logger.log(' upBEST bestscore for every population:%s' % str(self.BestScore_t_all.flatten()))
        self.BESTSCORE = np.max(self.BestScore_t_all.flatten()[1:])
        self.BESTSCORE_id = np.argmax(self.BestScore_t_all.flatten()[1:])+1
        self.BESTParam = self.BestParam_t_all[self.BESTSCORE_id].copy()
        if self.BESTSCORE > oldBESTSCORE and self.rank == 0:
            self.logger.save_parameters(self.BESTParam,self.iteration)

    def updateBest_t(self, score, param):
        """更新当前线程的最优个体得分与参数
        """
        if score > self.BestScore_t[0]:
            self.BestScore_t[0] = score
            self.BestParam_t = param.copy()

    def rollout(self, parameters):
        """对策略中的rollout的封装，支持k次评估
        返回值：
            msg: [mean_reward, sum_len]
                第一个为平均得分
                第二个为消耗的训练帧总和
        """
        lens = [0]
        rews = [0] 
        e_r = 0
        e_l = 0
        self.policy.set_parameters(parameters)
        for j in range(self.k):
            e_rew, e_len = self.policy.rollout()
            e_r += e_rew
            e_l += e_len
        rews[0] = e_r/self.k
        lens[0] = e_l
        msg = np.array(rews + lens)
        # print(msg)
        return msg

    def calLlambda(self, steps_passed):
        """计算 llambda的值，这里采用llambda表示算法中的lambda，因为lambda在python中是一个关键字
        """
        self.llambda = np.random.randn() * (0.1-0.1*steps_passed/self.steps_max) + 1.0

    def calphi(self,steps_passed):
        """自适应调整phi"""
        percent = steps_passed / self.steps_max
        return self.phi * (math.e - math.exp(percent)) / (math.e - 1)

    def calk(self,steps_passed):
        """自适应调整评估次数"""
        percent = steps_passed / self.steps_max
        k = int(self.k * percent)
        k = np.clip(k,1,10)
        return k

    def callr(self,steps_passed):
        """自适应调整lr mean 和lr sigma"""
        percent = steps_passed / self.steps_max
        lr_mean = self.lr_mean * (math.e - math.exp(percent)) / (math.e - 1)
        lr_sigma = self.lr_sigma * (math.e - math.exp(percent)) / (math.e - 1)
        return lr_mean,lr_sigma

    def logPath(self):
        """返回日志的路径
        """
        if self.args['env_type'] == 'atari':
            return "logs_mpi/%s/NCNES/%s/lam%s/mu%s/%s" %(self.args['game'],self.parallel, self.lam, self.pop_size, self.args['run_name'])
        elif self.args['env_type'] == 'function':
            return "logs_mpi/function%d/NCNES/%s/lam%s/mu%s/%s" %(self.args['function_id'],self.parallel, self.lam, self.pop_size, self.args['run_name'])

    def run(self):
        """算法类的运行函数，即主循环
        """
        self.start_time = time.time()
        steps_passed = 0

        # for retestBestFound
        self.last_retest_steps = 0
        self.logger.log('before main loop' + str(self.calDist(self.param)))

        while steps_passed <= self.steps_max:
            iter_start_time = time.time()
            steps_this_iter = 0
            self.calLlambda(steps_passed)
            # self.logger.log('upBEST bestscore for every population:%s' % str(self.BestScore_t_all.flatten()))
            self.logger.log('in main loop' + str(self.calDist(self.param)))

            # update 
            if self.lr_decay:
                self.lr_mean, self.lr_sigma = self.callr(steps_passed)

            if self.phi_decay:
                self.phi = self.calphi(steps_passed)

            if self.eva_decay:
                self.k = self.calk(steps_passed)
            self.logger.log('in main loop2' + str(self.calDist(self.param)))
            
            # different parallel mode
            if self.parallel == 'i':
                # generate child and evaluate it
                cost_steps = self.genAndEvalChild_i()
            elif self.parallel == 'p':
                self.logger.log(' main loop' + str(self.calDist(self.param)))

                cost_steps = self.genAndEvalChild_p()
            elif self.parallel == 's':
                cost_steps = self.genAndEvalChild_s()
            
            self.udpateBEST()
            self.params_all = self.syncOneVector(self.param)
            self.sigma_all = self.syncOneVector(self.sigma)
            steps_this_iter += cost_steps
            steps_passed += cost_steps

            params_grad,sigma_grad = self.calGrad(
                self.param,     self.sigma,
                self.param_all, self.sigma_all
            )
            self.updateMean(params_grad)
            self.updateSigma(sigma_grad)
            self.params_all = self.syncOneVector(self.param)
            self.sigma_all = self.syncOneVector(self.sigma)

            self.log(iter_start_time, steps_this_iter, steps_passed)
            self.iteration += 1
            self.retestBestFound(steps_passed)

    def retestBestFound(self, steps_passed):
        """重新测试保存的所有线程的最优解 self.cpus * k 次

        为了做到约50个点，采用last_retest_steps记录上一次进行
        重新测试的steps大小，当过去了约49w steps时，进行重新测
        试。采用49w是因为采用50w会导致测试点数便少，所有小一点用
        于补偿点数偏少。

        当种群大小N=6，k=7时，测试49次。
        """
        if steps_passed - self.last_retest_steps > 490000:
            self.last_retest_steps = steps_passed
            msg_new = self.rollout(self.BESTParam)
            reward = msg_new[0]
            reward_all = self.syncOneValue(reward)
            reward_mean = np.mean(reward_all)
            self.log_retest['steps'].append(steps_passed)
            self.log_retest['performance'].append(reward_mean)
            if self.rank == 0:
                self.logger.save_parameters(self.BESTParam, self.iteration)

    def save_retest_log(self):
        """保存重新测试的日志
        """
        filepath = os.path.join(self.logPath(), 'retest_log.pickle')
        with open(filepath, 'wb') as f:
            pickle.dump(self.log_retest, f)

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

    def genAndEvalChild_i(self):
        raise NotImplementedError

    def genAndEvalChild_s(self):
        raise NotImplementedError

    def genAndEvalChild_p(self):
        """产生子代并进行评估 = get reward

        非主线程都要产生一个子代，并进行评估，完成之
        后同步子代的适应度和消耗的训练帧数

        返回值：
            cost_steps ： 消耗的训练帧数
        """
        cost_steps = 0
        if self.rank != 0:
            # 生成子代
            # self.param_new = self.param + np.random.normal(scale = self.sigma,size = self.n)
            # self.param_new = np.clip(self.param_new,self.args['L'],self.args['H'])
            # * is element wise
            self.logger.log('params'+str(self.calDist(self.param)))
            self.param_new = self.param + np.random.normal(scale = self.sigma,size = self.n)
            self.param_new = np.clip(self.param_new,self.args['L'],self.args['H'])
            self.logger.log('paramsnew'+str(self.calDist(self.param_new)))
        
            
            # 评估子代
            msg_new = self.rollout(self.param_new)
            reward_child_t = msg_new[0]
            steps_cost_child = msg_new[1]
            self.updateBest_t(msg_new[0], self.param_new)
            self.logger.log('bestscore for every population:%d' % self.BestScore_t)
        else:
            # Empty array, evaluation results are not used for the update
            reward_child_t, reward_father_t = 0, 0
            steps_cost_father, steps_cost_child = 0, 0

        tmp_steps_cost = None
        # sync child reward
        self.reward_child = self.syncOneValue(reward_child_t)
        tmp_steps_cost = self.syncOneValue(steps_cost_child)
        cost_steps += np.sum(tmp_steps_cost)
        return cost_steps

    def log(self, iter_start_time, steps_this_iter, steps_passed):
        """日志函数
        """
        logger = self.logger
        if self.rank == 0:
            if self.iteration % 1000 == 0:
                logger.log("the best of iteration %d are %e" %(self.iteration, self.BESTSCORE))
            if self.args['env_type'] == 'atari':
                iteration_time = (time.time() - iter_start_time)
                time_elapsed = (time.time() - self.start_time)/60
                # train_mean_rew = np.mean(rews)
                # train_max_rew = np.max(rews)
                logger.log('------------------------------------')
                logger.log('Iteration'.ljust(25) + '%d' % self.iteration)
                logger.log('bestscore for every population:%s' % str(self.BestScore_t_all.flatten()))
                logger.log('StepsThisIter'.ljust(25) + '%f' % steps_this_iter)
                logger.log('StepsSinceStart'.ljust(25)+'%f' %steps_passed)
                logger.log('IterationTime'.ljust(25) + '%f' % iteration_time)
                logger.log('TimeSinceStart'.ljust(25) + '%d' %time_elapsed)
                logger.log('Best'.ljust(25) + '%f' % self.BESTSCORE)
            if self.iteration % 20 == 0:
                logger.save_parameters(self.BESTParam, self.iteration)
                self.save_retest_log()

    def saveAndTestBest(self):
        """运行算法之后，对最好的个体进行测试200次，并保存结果
        """
        logger = self.logger
        # print("save And Test best")
        # test best
        self.k = 1
        # 计算每个线程跑多少次游戏
        test_times = int(200 / self.cpus)
        # 剩余的给主线程来跑
        reminder = 200 - test_times * self.cpus
        final_rews = []
        for i in range(test_times):
            msg = self.rollout(self.BESTParam)
            final_rews.append(msg[0])
        part_one_reward = self.syncOneValue(np.mean(final_rews))

        if self.rank == 0:
            logger.log("t:%d" % test_times)
            logger.log("reminder:%d" % reminder)
            part_two_reward = []
            for i in range(reminder):
                msg = self.rollout(self.BESTParam)
                part_two_reward.append(msg[0])
            final_eval = (np.sum(part_two_reward) + np.sum(part_one_reward) * test_times) / 200

        if self.rank == 0:
            self.k = 1
            # final_rews = []
            # for i in range(200):
            #     msg = self.rollout(self.BESTParam)
            #     final_rews.append(msg[0])
            # final_eval = np.mean(final_rews)
            logger.log('Final'.ljust(25) + '%e' % final_eval)
            logger.save_parameters(self.BESTParam, self.iteration)
            time_elapsed = (time.time() - self.start_time)/60
            logger.log('TimeSinceStart'.ljust(25) + '%f' % time_elapsed)
            logger.log("random seed: %d" % self.randomSeed)
            self.save_retest_log()
   
    def logBasic(self):
        """基础信息的日志输出
        """
        logger = self.logger
        if self.rank == 0:
            logger.log("envtype:%s" % self.args['env_type'])
            logger.log("game:%s" % self.args['game'])
            logger.log("phi:%s" % self.phi)
            logger.log("lr sigma:%s" % self.lr_sigma)
            logger.log("lr mean:%s" % self.lr_mean)
            logger.log("N(Lam):%d" % self.lam)
            logger.log("pop Size(Mu):%d" % self.pop_size)
            logger.log("eva Time: %d" % self.k)
            logger.log("sigma0: %s" % str(self.sigma0))
            # logger.log("epoch: %d" % self.epoch)
            logger.log("r:%f" % self.r)
            logger.log("H: %d; L: %d" % (self.args['H'], self.args['L']))
            logger.log("stepMax:%d" % self.steps_max)
            logger.log("lr decay enable ?:%s" % self.lr_decay)
            logger.log("phi decay enable ?:%s" % self.phi_decay)

    def calUtility(self,rewards):
        rank = [0 for i in range(len(rewards))]
        for r,i in enumerate(np.argsort(rewards)[::-1]):
            rank[i] = r + 1  # rank kid by reward
        mu = self.pop_size
        util_ = np.maximum(0, np.log(mu / 2 + 1) - np.log(rank))
        utility = (util_ / (util_.sum())) - (1 / mu)
        return utility

    def calFFisher(self,params,sigma,params_list,sigma_list):
        '''计算fmean,fsigma,fishermean,fishersigma'''
        utility = self.calUtility(self.reward_child)

        fmean = np.zeros_like(params, dtype=np.float)
        fsigma = np.zeros_like(params, dtype=np.float)
        fishermean = np.zeros_like(params, dtype=np.float)
        fishersigma = np.zeros_like(params, dtype=np.float)

        # if param all is sync
        for i,params2 in enumerate(params_list):
            noise = params2 - params
            sigma_inverse = 1 / (sigma + self.eps)
            tmp1 = sigma_inverse*noise*noise*sigma_inverse
            tmp1 = np.clip(tmp1,self.args['L'],self.args['H'])
            tmp_ = tmp1 - sigma_inverse

            # print('utility %d' % len(utility))
            # print('pl %d' % len(params_list))
            fmean = fmean + sigma_inverse*noise*utility[i]
            fsigma = fsigma + tmp_*utility[i]
            fishermean = fishermean + tmp1
            fishersigma = fishersigma + np.clip(tmp_*tmp_,0.0001,1000)

        fmean = fmean/self.pop_size
        fsigma = fsigma/2/self.pop_size
        fishermean = fishermean/self.pop_size
        fishersigma = fishersigma/4/self.pop_size

        return fmean,fsigma,fishermean,fishersigma

    def calDiversity(self,params,sigma,params_list,sigma_list):
        '''计算dmean,dsigma'''

        dmean = np.zeros_like(params,dtype=np.float)
        dsigma = np.zeros_like(params,dtype=np.float)
    
        for params2, sigma2 in zip(params_list, sigma_list):
            sigma_part = 2 / (sigma + sigma2 + self.eps)
            params_minus = params - params2
            dmean = dmean + sigma_part * params_minus
            dsigma = dsigma + sigma_part-1/(4*sigma_part*params_minus*params_minus*sigma_part + self.eps)-1/(sigma + self.eps)

        dmean = dmean / 4
        dsigma = dsigma / 4

        return dmean,dsigma

    def calGrad(self,params,sigma,params_list,sigma_list):
        params_grad = np.zeros_like(params,dtype=np.float)
        sigma_grad = np.zeros_like(params,dtype=np.float)

        fmean,fsigma,fishermean,fishersigma = self.calFFisher(params,sigma,params_list,sigma_list)
        self.logger.log('fmean'+str(self.calDist(fmean)))
        self.logger.log('fsigma'+str(self.calDist(fsigma)))
        self.logger.log('sishermean'+str(self.calDist(fishermean)))
        self.logger.log('fishersigma'+str(self.calDist(fishersigma)))

        dmean,dsigma = self.calDiversity(params,sigma,params_list,sigma_list)
        self.logger.log('dmean'+str(self.calDist(dmean)))
        self.logger.log('dsigma'+str(self.calDist(dsigma)))


        params_grad = 1 / (fishermean * (fmean + self.phi * dmean) + self.eps)
        sigma_grad = 1 / (fishersigma * (fsigma + self.phi * dsigma) + self.eps)
        self.logger.log('params grad'+str(self.calDist(params_grad)))
        self.logger.log('sigma grad'+str(self.calDist(sigma_grad)))
        params_grad = self.checkbound(params_grad, -10, 10)
        sigma_grad = self.checkbound(sigma_grad, -10,10)

        return params_grad,sigma_grad
    
    def calDist(self,p):
        return np.max(p),np.min(p),np.mean(p),np.var(p)

    def updateSigma(self,sigma_grad):
        """更新协方差，并同步 
        """
        if self.rank != 0:
            self.sigma = self.sigma + self.lr_sigma * sigma_grad
            self.sigma = self.checkbound(self.sigma,1e-8,1e8)
        
    def updateMean(self,params_grad):
        """更新协方差，并同步 
        """
        if self.rank != 0:
            self.param = self.param + self.lr_mean * params_grad
            self.param = self.checkbound(self.param, self.args['L'], self.args['H'])
            # self.param = np.clip(self.param, self.args['L'], self.args['H'])
    
    @staticmethod
    def checkbound(params,low,high):
        params[np.isnan(params)]=np.mean(params)
        params = np.clip(params, low, high)
        return params
       
'''
NCNES算法的入口文件，用于处理输入参数。
输入参数：
    --run_name, -r    本次运行的别名，保存日志文件时会用到这个名字
    --env_type, -e    环境的类型（取值为 atari或者function）
    --game,-g         Atari游戏的名字，不需要带NoFrameskip-v4，代码中会自动加具体的版本信息
    --function_id, -f CEC测试函数的id，取值范围[1,2,3,4,5,6]
    --dimension, -d   CEC测试函数的维度
    --k, -k           演化过程中得到模型适应度值时模型被评估的次数
    --sigma0          NCNES算法中高斯噪声标准差的初始值
    --rvalue          NCNES算法中更新高斯噪声标准差的系数
    --lam             NCNES算法中种群数
    --mu              NCNES算法中种群中个体数
    --phi             NCNES算法中负相关系数
    --lr_sigma        NCNES算法中sigma梯度更新的学习率
    --lr_mean         NCNES算法中mean梯度更新的学习率
'''

@click.command()
@click.option('--run_name', '-r', required=True, type=click.STRING, help='Name of the run, used to create log folder name')
@click.option('--env_type', '-e', required=True, default = 'atari',type=click.Choice(['atari', 'function']))
@click.option('--game', '-g', type=click.STRING, help='game name for RL')
@click.option('--function_id', '-f', type=click.INT, help='function id for function optimization')
@click.option('--dimension', '-d', type=click.INT, help='the dimension of solution for function optimization')
@click.option('--k', '-k', type=click.INT, default=1, help='the number of evaluation times in training')
@click.option('--sigma0', type=click.FLOAT, default=2, help='the intial value of sigma')
@click.option('--rvalue', type=click.FLOAT, default=0.99, help='sigma update parameter')
@click.option('--lam', type=click.INT, default=5, help='population nums')
@click.option('--mu', type=click.INT, default=15, help='population size')
@click.option('--parallel','-p', type=click.STRING, default='p', help='parallel mode')
@click.option('--phi', type=click.FLOAT, default=0.00001, help='negative correation factor')
@click.option('--lr_sigma', type=click.FLOAT, default=0.2, help='sigma learning rate')
@click.option('--lr_mean', type=click.FLOAT, default=0.1, help='mean learning rate')
def main(run_name, env_type, game, function_id, dimension, k, sigma0, rvalue,parallel,phi,lam,mu,lr_sigma,lr_mean):
    # 算法入口
    kwargs = {
        'network': 'Nature',
        'nonlin_name': 'relu',
        'k': k,
        'sigma0': sigma0,
        'run_name': run_name,
        'env_type': env_type,
        'game': game,
        'function_id': function_id,
        'D': dimension,
        'r': rvalue,
        'H': 10,
        'L': -10,
        'parallel':parallel,
        'lr_sigma':lr_sigma,
        'lr_mean':lr_mean,
        'phi':phi,
        'lam':lam,
        'mu':mu
    }
    algo = NCNESAlgo(**kwargs)
    algo.run()
    # 最优解的最终测试以及保存日志
    algo.saveAndTestBest()
    

if __name__ == '__main__':
    main()

