

import time
import gym
import pickle
import os
import numpy as np

from mpi4py import MPI

from src.decomposer import Decomposer
from src.policy import Policy,FuncPolicy
from src.logger import Logger
from src.testfunc import test_func_bound

# 以下三个变量用于描述版本号和具体内容之间的联系
# 如版本1，3 采用Bestfound_i 来填充
VersionUsingBestfound = [1, 3]
VersionUsingFather = [2, 4]
VersionDivideEveryEpoch = [3, 4]


def calBdistance(param1, param2, sigma1, sigma2, mask_on):
    """计算分布之间的距离
    参数：
        param1(np.ndarray): 分布1的均值
        sigma1(np.ndarray): 分布1的协方差

        param2(np.ndarray): 分布2的均值
        sigma2(np.ndarray): 分布2的协方差

        mask_on(np.ndarray): 当前分组的mask，shape和param、sigma相同，取值为0或者1

    返回值：
        分布之间的距离值
    """
    xi_xj = param1 - param2
    big_sigma1 = sigma1 * sigma1
    big_sigma2 = sigma2 * sigma2
    big_sigma = (big_sigma1 + big_sigma2)/2
    small_value = 1e-8
    part1 = 1/8*np.sum(mask_on*xi_xj*xi_xj/(big_sigma+small_value))
    part2 = np.sum(np.log(big_sigma+small_value)*mask_on) - 1/2*np.sum(np.log(big_sigma1+small_value)*mask_on) - 1/2*np.sum(np.log(big_sigma2+small_value)*mask_on)
    return part1 + 1/2*part2


def calCorr(n, param_list, param, sigma_all, sigma, group_mask, rank_now, i_part):
    """计算分布param的相关性
    参数：
        n(int): the number of parameters
        
        param(np.ndarray): 当前分布的均值
        sigma(np.ndarray): 当前分布的协方差

        param_list(np.ndarray): 所有分布的均值
        sigma_all(np.ndarray): 所有分布的协方差

        group_mask(np.ndarray): 所有分组的mask，长度为n的向量，取值为[0,1,2,..,m-1]，m是分组的数量
        rank_now(int): 当前线程的id
        i_part(int): 当前分组的id
    返回值：
        这个分布的相关性
    """
    mask_on = (group_mask == i_part)
    assert len(sigma_all.shape)==2
    size = sigma_all.shape[0]
    DBlist = []
    for i in range(1, size):
        if i != rank_now:
            param2 = param_list[i]
            sigma2 = sigma_all[i]
            DB = calBdistance(param, param2, sigma, sigma2, mask_on)
            DBlist.append(DB)
    return np.min(DBlist)


class NCSCCAlgo(object):

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

            group_mask     分组方案的保存变量，为长度为n的向量，
                           取值为[0,1,2,3...m-1]，m为分组的数目
            group_number   分组的数目

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
        self.version = self.args['version']
        self.r = self.args['r']
        self.updateCount = 0
        

        if self.version in VersionDivideEveryEpoch:
            self.divide_every_iteration = self.epoch
        else:
            self.divide_every_iteration = 1

        self.randomSeed = np.random.randint(100000)
        self.rs = np.random.RandomState(self.randomSeed)
        if self.rank == 0:
            self.group_decomposer = Decomposer([2,3,4],self.randomSeed)
        self.logger = Logger(self.logPath())
        
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
        # 后者同步到前者

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

        self.train_set, self.test_set = self.getSeedPool(self.args['n_train'],self.args['n_test'],self.randomSeed)

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
        self.BESTSCORE = np.min(self.BestScore_t_all.flatten()[1:])
        self.BESTSCORE_id = np.argmin(self.BestScore_t_all.flatten()[1:])+1
        self.BESTParam = self.BestParam_t_all[self.BESTSCORE_id].copy()

    def updateBest_t(self, score, param):
        """更新当前线程的最优个体得分与参数
        """
        if score < self.BestScore_t[0]:
            self.BestScore_t[0] = score
            self.BestParam_t = param.copy()

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

    def calLlambda(self, steps_passed):
        """计算 llambda的值，这里采用llambda表示算法中的lambda，因为lambda在python中是一个关键字
        """
        self.llambda = self.rs.rand() * (0.1-0.1*steps_passed/self.steps_max) + 1.0

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

    def logPath(self):
        """返回日志的路径
        """
        if self.args['env_type'] == 'atari':
            return "logs_mpi/%s/NCSCC/lam%s/%s" %(self.args['game'], self.lam, self.args['run_name'])
        elif self.args['env_type'] == 'function':
            return "logs_mpi/function%s/NCSCC/lam%s/%s" %(self.args['function_id'], self.lam, self.args['run_name'])

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
            self.divideParameters()
            if (self.iteration -1) % self.divide_every_iteration == 0:
                # 每epoch代的第一代需要对更新次数统计变量self.updateCount进行清0
                self.updateCount = np.zeros(self.n, dtype=np.int32)
            for group_id in range(self.group_number):
                # generate child and evaluate it
                cost_steps = self.generateAndEvalChild(group_id)
                self.udpateBEST()
                self.replaceFather(group_id)
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
            msg_new = self.rollout(self.BESTParam,test=True)
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

    def generateAndEvalChild(self, group_id):
        """产生子代并进行评估

        非主线程都要产生一个子代，并进行评估，完成之
        后同步子代的适应度和消耗的训练帧数

        参数：
            group_id : 当前的分组id

        返回值：
            cost_steps ： 消耗的训练帧数
        """
        cost_steps = 0
        if self.rank != 0:
            # refer to NCSCC pseudo code line 11
            # 优化当前分组group_id时，其余分组的参数值采用当前保留的最优解的参数
            if self.version in VersionUsingBestfound:
                param_newpart1 = self.param.copy() + self.rs.randn(self.n) * self.sigma
                self.param_new = param_newpart1 * (self.group_mask==group_id) + self.BestParam_t * (self.group_mask!=group_id)
            else:
                self.param_new = self.param.copy() + self.rs.randn(self.n) * self.sigma * (self.group_mask == group_id)

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
        self.reward_child = self.syncOneValue(reward_child_t)

        if self.version in VersionUsingBestfound:
            # sync father reward
            self.reward_father = self.syncOneValue(reward_father_t)
            tmp_steps_cost = self.syncOneValue(steps_cost_father)
            cost_steps += np.sum(tmp_steps_cost)

        tmp_steps_cost = self.syncOneValue(steps_cost_child)
        cost_steps += np.sum(tmp_steps_cost)
        return cost_steps

    def replaceFather(self, group_id):
        """替换父代

        根据算法计算是否用子代替换父代，只有非主线程参与
        需要做的事情：
            1. 适应度归一化
            2. 计算相关性，并归一化
            3. 是否替换
            4. 是否需要更新父代个体的适应度值
            5. 同步父代的参数

        参数：
            group_id :   当前分组id
        """
        if self.rank != 0:
            # refer to NCSCC pseudo code line 12
            father_corr = calCorr(self.n, self.param_all, self.param, self.sigma_all, self.sigma, self.group_mask, self.rank, group_id)
            child_corr = calCorr(self.n, self.param_all, self.param_new, self.sigma_all, self.sigma, self.group_mask, self.rank, group_id)
            # 每个线程计算自己的correlation和new correlation， 但是对于相关性和fitness都需要进行归一化
            child_corr = child_corr / (father_corr + child_corr)
            # 优化目标是到最小
            father_f = self.reward_father[self.rank] - self.BESTSCORE + 10**-10
            child_f = self.reward_child[self.rank] - self.BESTSCORE + 10**-10
            child_f = child_f / (child_f + father_f)
            # refer to NCSCC pseudo code line 19
            if child_f / child_corr < self.llambda:
                # 抛弃旧的解，更换为新解
                self.param[self.group_mask == group_id] = self.param_new[self.group_mask == group_id].copy()
                self.updateCount = self.updateCount + (self.group_mask == group_id)
                self.reward_father[self.rank] = self.reward_child[self.rank]
            reward_father_t = self.reward_father[self.rank]
        else:
            reward_father_t = 0
        # 更新父代的适应度值
        if self.version not in VersionUsingBestfound:
            self.reward_father = self.syncOneValue(reward_father_t)
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
                logger.log('the number of groups:%d' % self.group_number)
                logger.log('bestscore for every population:%s' % str(self.BestScore_t_all))
                # logger.log('EvalMeanReward'.ljust(25) + '%f' % eval_mean_rew)
                # logger.log('EvalMeanReward1'.ljust(25) + '%f' % eval_mean_rew1)
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
            msg = self.rollout(self.BESTParam,test=True)
            final_rews.append(msg[0])
        part_one_reward = self.syncOneValue(np.mean(final_rews))

        if self.rank == 0:
            logger.log("t:%d" % test_times)
            logger.log("reminder:%d" % reminder)
            part_two_reward = []
            for i in range(reminder):
                msg = self.rollout(self.BESTParam,test=True)
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
            logger.log("S:%s" % str(self.group_decomposer.condidate_size))
            logger.log("stepMax:%d" % self.steps_max)



"""
NCSCC算法的入口文件，用于处理输入参数。
输入参数：
    --version, -v     算法的版本
    --run_name, -r    本次运行的别名，保存日志文件时会用到这个名字
    --env_type, -e    环境的类型（取值为 atari或者function）
    --game,-g         Atari游戏的名字，不需要带NoFrameskip-v4，代码中会自动加具体的版本信息
    --function_id, -f CEC测试函数的id，取值范围[1,2,3,4,5,6]
    --dimension, -d   CEC测试函数的维度
    --k, -k           演化过程中得到模型适应度值时模型被评估的次数
    --epoch,          NCSCC算法中更新高斯噪声标准差周期，一般取值为5的倍数
    --sigma0,         NCSCC算法中高斯噪声标准差的初始值
    --rvalue          NCSCC算法中更新高斯噪声标准差的系数


NCSCC一共四个版本，现在选择版本4作为最终的算法版本，可以设置--version参数的默认值为4，也可以在输入参数中加入 -v 4。
版本1: 
1.	采用Bestfound_i 来填充
2.	每一代重新分组
版本2:
1.	采用父代中除去子问题i以外的参数
2.	每一代重新分组
版本3: 
1.	采用Bestfound_i 来填充
2.	每epoch代重新分组
版本4:
1.	采用父代中除去子问题i以外的参数
2.	每epoch代重新分组
"""

@click.command()
@click.option('--version', '-v', required=True, default=1, type=click.INT, help='the version of algorithm')
@click.option('--run_name', '-r', required=True, type=click.STRING, help='Name of the run, used to create log folder name')
@click.option('--env_tyep', '-e', required=True, type=click.Choice(['atari', 'function']))
@click.option('--game', '-g', type=click.STRING, help='game name for RL')
@click.option('--function_id', '-f', type=click.INT, help='function id for function optimization')
@click.option('--dimension', '-d', type=click.INT, help='the dimension of solution for function optimization')
@click.option('--k', '-k', type=click.INT, default=1, help='the number of evaluation times in training')
@click.option('--epoch', type=click.INT, default=10, help='the number of epochs updating sigma')
@click.option('--sigma0', type=click.FLOAT, default=0.01, help='the intial value of sigma')
@click.option('--rvalue', type=click.FLOAT, default=0.99, help='sigma update parameter')
def main(version, run_name, env_tyep, game, function_id, dimension, k, epoch, sigma0, rvalue):
    # 算法入口
    kwargs = {
        'network': 'Nature',
        'nonlin_name': 'relu',
        'k': k,
        'epoch': epoch,
        'sigma0': sigma0,
        'version': version,
        'run_name': run_name,
        'env_type': env_tyep,
        'game': game,
        'function_id': function_id,
        'D': dimension,
        'r': rvalue,
        'n_test': 10000,
        'n_train': 10000
    }
    algo = NCSCCAlgo(**kwargs)
    algo.run()
    # 最优解的最终测试以及保存日志
    algo.saveAndTestBest()
    

if __name__ == '__main__':
    main()
