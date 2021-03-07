
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo.py
@Time    :   2020/06/27 16:41:31
@Author  :   Qi Yang
@Describtion:  定义模仿学习 demo 类和读取 demo 缓存类

datastructure 
demo dict
- actions
- signed rewards
- reward
- lives
- checkpoints (observations)
- checkpoint_action_nr
demo info
- idx              index in demo
- increase_entropy flag
- as good as demo  flag
- action_nr        action_nr
- valid            flag
- live             live
- score            return
'''
import pickle
import numpy as np

class Demo:
    r'''
    Huamn demostration class including action reward obs lives in a trajectory.  
    Args:  
        actions = None
        rewards = None
        returns = [0]
        obs = None
        lives = None
        action_nr = None
        length = 0
        info = {}
    Methods:  
        replay(i): replay infomations at index i.  
    '''
    def __init__(self,demo_file_name):
        assert demo_file_name is not None
        if demo_file_name is None:
            self.actions = None
            # self.log_prob = None
            self.rewards = None
            self.returns = [0]
            self.obs = None
            self.lives = None
            self.action_nr = None
            self.length = 0
            self.point = 0
        else:
            with open(demo_file_name, "rb") as f:
                dat = pickle.load(f)
                self.actions = dat['actions']
                self.rewards = dat['rewards']
                self.length = len(self.actions)
                self.returns = np.cumsum(self.rewards)
                assert len(self.rewards) == len(self.actions)
                self.lives = dat['lives']
                self.obs = dat['checkpoints']
                self.action_nr = dat['checkpoint_action_nr']
                self.point = 0

    def act(self):
        i = self.point
        self.point += 1
        return self.actions[i]

    def step(self):
        """return action, ob, reward, done, info in demo.  
        info example:
        - idx:1  
        - done:False 
        - lives: 1
        - score: 1
        """
        i = self.point
        idx = int(i/100)
        info = {}
        if self.lives[i] == 0:
            done = True
        else:
            done = False
        info['episode_info'] = {
            'lives':self.lives[i],
            'score':self.returns[idx],
            'action_nr':self.action_nr[idx],
        }
        info['idx'] = i
        info['achived_done'] = False
        self.point += 1
        return self.obs[idx],self.rewards[idx],done,info

class Buffer_():
    def __init__(self):
        self.actions = []
        self.obs = []
        self.rewards = []
        self.dones = []
        self.size = 0
        self.is_trains = []
        self.infos = []
        self.state_value = []
        self.discount_r = []
    
    def clear(self):
        self.actions, self.obs, self.rewards, self.dones = [],[],[],[]
        self.infos, self.is_trains, self.state_value, self.discount_r = [],[],[],[]
        
    def add(self,action,ob,reward,done,info,is_train):
        self.actions.append(action)
        self.obs.append(ob)
        self.dones.append(done)
        self.is_trains.append(is_train)
        self.rewards.append(reward)
        self.infos.append(info)
        self.size = len(self.rewards)

    def get_size(self):
        assert len(self.rewards) == len(self.actions)
        return self.size                                             

class Buffer():
    def __init__(self):
        self.data = []
        self.point = 0
        self.length = len(self.data)
    
    def clear(self):
        self.data = []
        self.point = 0
        self.length = len(self.data)
        
    def add(self,action,ob,reward,done,info):
        self.data.append((action,ob,reward,done,info))
        self.length += 1
    
    def act(self,i):
        return self.data[i][0]

    def step(self,i):
        return self.data[i][1:]


