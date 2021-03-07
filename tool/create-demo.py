#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   create-demo.py
@Time    :   2020/09/02 14:02:45
@Author  :   Qi Yang
@Version :   1.0
@Describtion:  创建 demo.py 定义的人类演示文件 Buf 
'''

# Code refence : https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py
#!/usr/bin/env python
import sys, gym, time
import pickle
from src.demo import Buffer
from PIL import Image
import numpy as np



#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# xvfb-run -s "-screen 0 1400x900x24" python create-demo.py SpaceInvadersNoFrameskip-v4
#

env = gym.make('LunarLander-v2' if len(sys.argv)<2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    
# Use previous control decision SKIP_CONTROL times, that's how you
# can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==27: # ESC
        human_wants_restart = True
    if key==32: # spacebar 
        human_sets_pause = not human_sets_pause
    keyvalue = int( key - ord('a'))
    if keyvalue < 0 or keyvalue > 26:
        a = 0
    else:
        a = keyboard[keyvalue]
        print(a)
    if a <= 0 or a >= ACTIONS: 
        return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    keyvalue = int( key - ord('a') )
    if keyvalue < 0 or keyvalue > 26:
        a = 0
    else:
        a = keyboard[keyvalue]
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    buf = Buffer()
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        
        #img = Image.fromarray(obser, 'RGB')
        #img.save('my.png')
        buf.add(a,obser,r,done,info)
        if r != 0 or total_timesteps%100==0:
            print("timesteps %i reward %0.3f" % (total_timesteps,r))
        if total_timesteps > 1000:
            with open('./demos/'+sys.argv[1]+'-1000.pickle','wb+') as f:
                pickle.dump(buf,f)
        total_reward += r
        window_still_open = env.render() 
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

# keyboard[key] = act
keyboard = [4,17,8,3,6,0,12,0,0,11,0,0,16,13,0,0,7,0,1,15,14,0,2,5,10,9]

ACTION_MEANING = {
    'h': "NOOP",
    's': "FIRE",
    'w': "UP",
    'd': "RIGHT",
    'a': "LEFT",
    'x': "DOWN",
    'e': "UPRIGHT",
    'q': "UPLEFT",
    'c': "DOWNRIGHT",
    'z': "DOWNLEFT",
    'y': "UPFIRE",
    'j': "RIGHTFIRE",
    'g': "LEFTFIRE",
    'n': "DOWNFIRE",
    'u': "UPRIGHTFIRE",
    't': "UPLEFTFIRE",
    'm': "DOWNRIGHTFIRE",
    'b': "DOWNLEFTFIRE",
}
print("ACTIONS={}".format(ACTIONS))
print("Move Direction is controlled by S-oritented keys while Move+Fire Directions by H-oritented")
print("No keys pressed is taking action 0")
print("ACTION MEANS = {}".format(ACTION_MEANING))
while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break

