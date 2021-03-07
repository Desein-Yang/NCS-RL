# build model for trained parameters 
import sys,os
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import pickle
import gym
import numpy as np 
from src.policy import Policy
from src.logger import Logger
from src.env_wrappers import wrap_dqn,NoopResetEnv,MaxAndSkipEnv,ProcessFrame84,FrameStack,FireResetEnvLife
import matplotlib.pyplot as plt
import seaborn as sns
import atari_py as ap
import multiprocessing as mp

class FixNoopResetEnv(gym.Wrapper):
    def __init__(self,env, fix_noop,  noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        支持三种形式的noop，从列表中选择、0-30 或固定值
        """
        super(FixNoopResetEnv, self).__init__(env)
        self.fix_noop = fix_noop
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            if self.fix_noop is None:
                noops = np.random.randint(1, self.noop_max + 1) 
            elif type(self.fix_noop) == list:
                noops = np.random.choice(self.fix_noop)
            else:
                noops = self.fix_noop
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

def wrapper(env,noop=None):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    # env = EpisodicLifeEnv(env)
    env = FixNoopResetEnv(env,fix_noop=noop)
    fs = 4
    if "SpaceInvaders" in env.spec.id:
        fs = 3
    env = MaxAndSkipEnv(env, skip=fs)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnvLife(env)
    env = ProcessFrame84(env)
    env = FrameStack(env, 4)
    # env = ClippedRewardsWrapper(env)
    return env

class EnvAnalysis(object):
    def __init__(self,game,env):
        self.game = game
        self.model_path = './logs/'+self.game


        env = wrapper(env,None)
        self.policy = Policy(env, network='Nature', nonlin_name='relu')
        vb = self.policy.get_vb()
        self.policy.set_vb(vb)
        self.logpath = './logs/'+self.game
        self.logger = Logger(self.logpath)
        if os.path.exists(os.path.join(self.model_path,self.game)):
            param = self.logger.load_parameters(os.path.join(self.model_path,self.game))
            self.policy.set_parameters(param)

    def randomExp(self,runname):
        """测试确定性agent对随机种子敏感程度（初始帧固定）"""
        k = 5 
        group = 30
        fix_noop = np.random.randint(1,31)
        logpath = './logs_randseed/'+self.game
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        self.logger.set_logdir(logpath)
        self.logger.log('random seed sensitivity'.ljust(25))
        self.logger.log('fix noop:%s'.ljust(25)  % str(fix_noop))
        self.logger.log('eva times:%s'.ljust(25) % str(k))
        self.logger.log('games:%s'.ljust(25) % self.game)
        # self.logger.log('model:%s'.ljust(25) % self.model_path+self.model_name)

        rew_noops, step_noops, aver_rew_noops,aver_step_noops,var_rew_noops = [],[],[],[],[]
        similarity = []
        randomseeds = [np.random.randint(100000) for i in range(group)]
        for seed in randomseeds:
            rew_k,step_k,av_rew,av_step = self.rollout(seed=seed,fixnoop=fix_noop,k=k)
            rew_noops.append(rew_k)
            step_noops.append(step_k)
            aver_rew_noops.append(av_rew)
            aver_step_noops.append(av_step)
            var_rew_noops.append(np.var(rew_k))

        self.logger.log('rew of %d group %s'%(group,str(rew_noops)))
        self.logger.log('step of %d group %s'%(group,str(step_noops)))
        self.logger.log('average rew of %d  %s'%(k,str(aver_rew_noops)))
        self.logger.log('var of rew %d  %s'%(k,str(var_rew_noops)))
        self.logger.log('average step of %d  %s'%(k,str(aver_step_noops)))
        self.logger.log('random seed %s'%str(randomseeds))

        for j in range(group):
            sim = [self.countSame(rew_noops[j],rew_noops[i])/k for i in range(group)]
            self.logger.log('similarity of seeds %d: %s' %(j, str(sim)))
            similarity.append(sim)
        self.drawHeatmap(similarity,'rand')

    def calSimilar(self,rews):
        def countSame2(array1,array2):
            assert len(array1) == len(array2)
            count = 0
            array1 = np.sort(array1)
            array2 = np.sort(array2)
            for i,j in zip(array1,array2):
                if i == j:
                    count += 1
            return count
        similarity = []
        for rew_k_i in rews:
            sim = [countSame2(rew_k_i,rew_k_j)/len(rew_k_i) for rew_k_j in rews]
            similarity.append(sim)
        return similarity

    def noopExp(self,runname):
        """测试确定性agent对不同初始帧敏感程度（随机种子固定）"""
        # rew_sum,frame = policy.rollout()
        seed = np.random.randint(100000)
        k = 30
        group = 5
        self.logger.set_logdir('./logs_noop/'+runname+'/'+self.game)

        self.logger.log('seed:%s'.ljust(25)      % str(seed))
        self.logger.log('eva times:%s'.ljust(25) % str(k))
        self.logger.log('games:%s'.ljust(25) % self.game)
        # self.logger.log('model:%s'.ljust(25) % self.model_path+self.model_name)

        rew_noops, step_noops, aver_rew_noops,aver_step_noops,var_rew_noops = [],[],[],[],[]
        similarity = []
        fixnooops = []
        for i in range(1,31,6):
            rew_k,step_k,a_rew,a_step = self.rollout(None, seed,fixnoop=i,k=k)
            rew_noops.append(rew_k)
            step_noops.append(step_k)
            aver_rew_noops.append(a_rew)
            aver_step_noops.append(a_step)
            var_rew_noops.append(np.var(rew_k))
            fixnooops.append(i)
        
        # self.saveLog(rew_noops,step_noops)
        for i in range(group):
            self.logger.log('rew of 5 group %s'%str(rew_noops[i]))
            self.logger.log('step of 5 group %s'%str(step_noops[i]))
        self.logger.log('average rew of 30 noop %s'%str(aver_rew_noops))
        self.logger.log('average var of 30 noop %s'%str(var_rew_noops))
        self.logger.log('average step of 30 noop %s'%str(aver_step_noops))
        self.logger.log('fix noops %s'%str(fixnooops))

        for j in range(group):
            sim = [self.countSame(rew_noops[j],rew_noops[i])/k for i in range(group)]
            self.logger.log('similarity of frame %d: %s' %(j, str(sim)))
            similarity.append(sim)
        self.drawHeatmap(similarity,'noop')

    def drawHeatmap(self,similarity,expname):
        similarity = np.array(similarity)
        fig, ax = plt.subplots()
        sns.heatmap(similarity, ax=ax)
        
        
        if expname == 'rand':
            ax.set_title('Different randomseed similarity')
            ax.set_xlabel('random seed group id')
            ax.set_ylabel('random seed group id')
            fig.savefig('./logs_randseed/'+self.game+'/rand_heatmap.png')
        else:
            ax.set_title('Different start frame similarity')
            ax.set_xlabel('start frame')
            ax.set_ylabel('start frame')
            fig.savefig('./logs_noop/'+self.game+'/'+expname +'_heatmap.png')

    def saveLog(self,rew_noops,step_noops):
        log = {
            'rew':rew_noops,
            'step':step_noops
        }
        with open(self.logpath+'/noop-rew.pickle','wb') as f:
            pickle.dump(log,f)
    
    def loadLog(self):
        with open(self.game+'-noop-rew.pickle','rb') as f:
            log = pickle.load(f)
            import pdb; pdb.set_trace()

    def countSame(self,array1,array2):
        """计算相同元素个数"""
        count = 0
        for i in array1:
            if i in array2:
                count += 1 
        return count
        # return len(set(array1) & set(array2))        
    
    def multirollout(self,cpus=10,param=None,seed=None,fixnoop=None,k =30):
        """rollout 随机版本"""
        pool = mp.Pool(cpus)
        results = []
        rew_k, step_k= [],[]
        for i in range(k):
            results.append(pool.apply_async(self.rollout,(seed,fix_noop,1,)))
        pool.close()
        pool.join()
        for res in results:
            rew_k.append(res.get()[0])
            step_k.append(res.get()[1])
            
        average_rew = np.mean(rew)
        average_step = np.mean(step)
        return rew_k,step_k,average_rew,average_step

    def rollout(self,param=None,seed=None,fixnoop=None,k=30):
        """在指定帧和指定随机种子下测试30次的结果返回  
        fixnoop,seed=None时则默认30noop和无随机种子"""
        env = gym.make("%sNoFrameskip-v4" % self.game)
        env = wrapper(env,fixnoop)
        env.seed(seed)
        self.policy.env = env 
        if param is not None:
            self.policy.set_parameters(param)
        
        rew_k = []
        step_k = []
        for i in range(k):
            rew, step = self.policy.rollout()
            rew_k.append(rew)
            step_k.append(step)
        
        average_rew = np.mean(rew)
        average_step = np.mean(step)
        return rew_k,step_k,average_rew,average_step
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game','-g',default=None)
    args = parser.parse_args()
    game = args.game
    try:
        env = gym.make("%sNoFrameskip-v4" % game)
    except:
        pass
    else:
        exp = EnvAnalysis(game,env)
        exp.noopExp()

# print(exp.rollout(seed=12346)

# ([185.0, 70.0, 185.0, 100.0, 70.0, 100.0, 70.0, 100.0, 185.0, 185.0, 70.0, 100.0, 100.0, 70.0, 185.0, 100.0, 70.0, 70.0, 185.0, 185.0, 70.0, 185.0, 70.0, 70.0, 185.0, 100.0, 100.0, 70.0, 70.0, 100.0], [968, 525, 967, 399, 528, 398, 531, 403, 966, 965, 525, 402, 396, 531, 962, 402, 528, 525, 961, 970, 527, 970, 524, 530, 967, 402, 395, 524, 530, 397], 100.0, 397.0)
# 同一个随机种子不同的随机帧居然跑的不一样？？