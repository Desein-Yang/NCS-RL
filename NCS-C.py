
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2020/06/27 16:41:31
@Author  :   Qi Yang
@Describtion:  NCS-C 算法(直接实现NCS-C类)
'''

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


class NCSAlgo(object):
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
        self.lam = self.cpus - 1
        self.k = self.args['k']
        self.epoch = self.args['epoch']
        self.sigma0 = self.args['sigma0']
        self.r = self.args['r']
        self.updateCount = 0


        self.randomSeed = np.random.randint(100000)
        self.rs = np.random.RandomState(self.randomSeed)
        self.logger = Logger(self.logPath())

        self.train_set, self.test_set = self.getSeedPool(self.args['n_train'],self.args['n_test'],self.randomSeed)        
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
        msg = self.rollout(self.param)
        results = np.empty((self.cpus, 2))
        self.comm.Allgather([msg, MPI.DOUBLE], [results, MPI.DOUBLE])
        self.reward_father = results[:,:1].flatten()
        self.BestScore_t[0] = msg[0]
        self.comm.Allgather([self.BestScore_t, MPI.DOUBLE], [self.BestScore_t_all, MPI.DOUBLE])
        self.udpateBEST()

    def udpateBEST(self):
        """依据每个线程中的最优个体来得到所有线程中的最优个体
        """
        self.comm.Allgather([self.BestParam_t, MPI.DOUBLE], [self.BestParam_t_all, MPI.DOUBLE])
        self.comm.Allgather([self.BestScore_t, MPI.DOUBLE], [self.BestScore_t_all, MPI.DOUBLE])
        self.BESTSCORE = np.max(self.BestScore_t_all.flatten()[1:])
        self.BESTSCORE_id = np.argmax(self.BestScore_t_all.flatten()[1:])+1
        self.BESTParam = self.BestParam_t_all[self.BESTSCORE_id].copy()

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
        print(msg)
        return msg

    def calLlambda(self, steps_passed):
        """计算 llambda的值，这里采用llambda表示算法中的lambda，因为lambda在python中是一个关键字
        """
        self.llambda = self.rs.rand() * (0.1-0.1*steps_passed/self.steps_max) + 1.0

    def logPath(self):
        """返回日志的路径
        """
        if self.args['env_type'] == 'atari':
            return "logs_mpi/%s/NCS/lam%s/%s" %(self.args['game'], self.lam, self.args['run_name'])
        elif self.args['env_type'] == 'function':
            return "logs_mpi/function%s/NCS/lam%s/%s" %(self.args['function_id'], self.lam, self.args['run_name'])

    def run(self):
        """算法类的运行函数，即主循环
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
            if self.iteration -1 == 0:
                # 每epoch代的第一代需要对更新次数统计变量self.updateCount进行清0
                self.updateCount = np.zeros(self.n, dtype=np.int32)

            # generate child and evaluate it
            cost_steps = self.generateAndEvalChild()
            self.udpateBEST()
            self.replaceFather()
            steps_this_iter += cost_steps
            steps_passed += cost_steps

            self.updateSigma()
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

    def generateAndEvalChild(self):
        """产生子代并进行评估

        非主线程都要产生一个子代，并进行评估，完成之
        后同步子代的适应度和消耗的训练帧数

        返回值：
            cost_steps ： 消耗的训练帧数
        """
        cost_steps = 0
        if self.rank != 0:
            # 生成子代
            self.param_new = self.param + self.rs.normal(scale = self.sigma,size = self.n)

            # 评估子代
            msg_new = self.rollout(self.param_new)
            reward_child_t = msg_new[0]
            steps_cost_child = msg_new[1]
            self.updateBest_t(msg_new[0], self.param_new)
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

    def replaceFather(self):
        """替换父代

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
        self.comm.Allgather([self.param, MPI.DOUBLE], [self.param_all, MPI.DOUBLE])

    def updateSigma(self):
        """更新协方差，并同步 
        """
        if self.rank != 0:
            if self.iteration % self.epoch == 0:
                self.sigma[self.updateCount/self.epoch<0.2] = self.sigma[self.updateCount/self.epoch<0.2] * self.r
                self.sigma[self.updateCount/self.epoch>0.2] = self.sigma[self.updateCount/self.epoch>0.2] / self.r
        # if self.iteration % self.epoch == 0:
        self.comm.Allgather([self.sigma, MPI.DOUBLE], [self.sigma_all, MPI.DOUBLE])

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
            logger.log("N:%d" % self.lam)
            logger.log("k: %d" % self.k)
            logger.log("sigma0: %s" % str(self.sigma0))
            logger.log("epoch: %d" % self.epoch)
            logger.log("r:%f" % self.r)
            logger.log("stepMax:%d" % self.steps_max)

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


'''
NCS算法的入口文件，用于处理输入参数。
输入参数：
    --run_name, -r    本次运行的别名，保存日志文件时会用到这个名字
    --env_type, -e    环境的类型（取值为 atari或者function）
    --game,-g         Atari游戏的名字，不需要带NoFrameskip-v4，代码中会自动加具体的版本信息
    --function_id, -f CEC测试函数的id，取值范围[1,2,3,4,5,6]
    --dimension, -d   CEC测试函数的维度
    --k, -k           演化过程中得到模型适应度值时模型被评估的次数
    --epoch,          NCS算法中更新高斯噪声标准差周期，一般取值为5的倍数
    --sigma0,         NCS算法中高斯噪声标准差的初始值
    --rvalue          NCS算法中更新高斯噪声标准差的系数
'''

@click.command()
@click.option('--run_name', '-r', required=True, default = 'debug',type=click.STRING, help='Name of the run, used to create log folder name')
@click.option('--env_type', '-e', required=True, default = 'atari',type=click.Choice(['atari', 'function']))
@click.option('--game', '-g', type=click.STRING, default = 'Alien', help='game name for RL')
@click.option('--function_id', '-f', type=click.INT, help='function id for function optimization')
@click.option('--dimension', '-d', type=click.INT, help='the dimension of solution for function optimization')
@click.option('--k', '-k', type=click.INT, default=10, help='the number of evaluation times in training')
@click.option('--epoch', type=click.INT, default=10, help='the number of epochs updating sigma')
@click.option('--sigma0', type=click.FLOAT, default=0.2, help='the intial value of sigma')
@click.option('--rvalue', type=click.FLOAT, default=0.8, help='sigma update parameter')
def main(run_name, env_type, game, function_id, dimension, k, epoch, sigma0, rvalue):
    # 算法入口
    kwargs = {
        'network': 'Nature',
        'nonlin_name': 'relu',
        'k': k,
        'epoch': epoch,
        'sigma0': sigma0,
        'run_name': run_name,
        'env_type': env_type,
        'game': game,
        'function_id': function_id,
        'D': dimension,
        'r': rvalue,
        'n_test': 10000,
        'n_train': 10000        
    }
    algo = NCSAlgo(**kwargs)
    algo.run()
    # 最优解的最终测试以及保存日志
    algo.saveAndTestBest()
    

if __name__ == '__main__':
    main()

