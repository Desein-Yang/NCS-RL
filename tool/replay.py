import gym
from src.wrapper import wrap_deepmind
from src.demo import Demo,Buffer
import numpy as np
from moviepy.editor import ImageSequenceClip

class ReplayAlgo():
    '''把demo里的动作重演一遍，保存每一个画面
    '''
    def __init__(self):
        self.FRAMESKIP = 4
        
        self.demo = Demo('./demos/Pitfall_origin.demo')
        print('demo action length %d' % len(self.demo.actions))
        print('demo observation %d' % len(self.demo.obs))
        self.TIMELIMIT = len(self.demo.actions)
        # self.actions[i],self.obs[idx],self.rewards[idx],done,info
        # print('demo ' % len(demo))

        self.env = gym.make('PitfallNoFrameskip-v4')
        # self.env = gym.make('PitfallDeterministic-v0')
        self.env = wrap_deepmind(self.env,frame_stack=4)

        self.humanBuf = Buffer() # my demo

    def rollout(self, render=False):
        # Evaluates the policy for up to max_episode_len steps.
        ob = self.env.reset()
        print('Start!')
        ob = np.asarray(ob)
        t = 0
        rew_sum = 0
        flag = False
        for _ in range(self.TIMELIMIT):
            ac = self.demo.act()
            ob_, rew_, done_, info_ = self.env.step(ac)
            ob_ = np.asarray(ob_)
            rew_sum += rew_
            self.humanBuf.add(ac,ob_,rew_,done_,info_)
            # ob,rew,done,info = self.demo.
            # self.humanBuf.add(ac,ob,rew,done,info)
            # t += 1
            # if render:
            #     self.env.render()
            if t % 100 == 0:
                print('Framecount %d reward %f' % (t, rew_sum))
                if flag == True:
                    break
            if rew_ > 0:
                print('Non-zero reward!')
                flag = True
            if done_:
                break

    def save(self,name):
        '''保存图像，记录视频'''
        import pickle

        with open('human_buf.pickle','wb+') as f:
            pickle.dump(self.humanBuf,f)
        # with open('env_buf.pickle','wb+') as f:
        #     pickle.dump(self.envBuf,f)

        obs_list = self.env.env.env.obs_list
        clip = ImageSequenceClip(obs_list, fps=24)
        clip.write_videofile(name, fps=24)


# replay
alg = ReplayAlgo()
alg.rollout(render=True)
alg.save("Pitfall_replay.mp4")
# # import pdb; pdb.set_trace()


# 读取buf查看动作
import pickle
action = []
with open('human_buf.pickle','rb+') as f:
    humanBuf = pickle.load(f)
for i in humanBuf.data:
    action.append(i[0])


