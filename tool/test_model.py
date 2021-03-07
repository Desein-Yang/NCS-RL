
import time
import gym
import pickle
import os
import click
import numpy as np

from mpi4py import MPI
from decomposer import Decomposer
from src.policy import SupervisePolicy
from src.logger import Logger
from src.wrapper import wrap_deepmind
from moviepy.editor import ImageSequenceClip
# from NCSCC_pitfall import NCSCCAlgo

# @click.command()
# @click.option('--run_name', '-r', required=True, type=click.STRING, help='Name of the run, used to create log folder name')
# @click.option('--env_type', '-e', required=True, default = 'atari',type=click.Choice(['atari', 'function']))
# @click.option('--game', '-g', type=click.STRING, help='game name for RL')
# @click.option('--epoch', type=click.INT, default=10, help='the number of epochs updating sigma')
# @click.option('--sigma0', type=click.FLOAT, default=1, help='the intial value of sigma')
# @click.option('--rvalue', type=click.FLOAT, default=0.99, help='sigma update parameter')
# @click.option('--steps_max','-s', type=click.INT, default=1e9,help='timesteps limit')
# def main(run_name, env_type, game, epoch, sigma0, rvalue, steps_max):
#     # kwargs = {
#     #     'network': 'Nature',
#     #     'nonlin_name': 'relu',
#     #     'epoch': epoch,
#     #     'k':1,
#     #     'loadpath':'/home/erl/LearnDemo2/logs_mpi/Pitfall/NCSCC/lam8/short/parameters_1722',
#     #     'version':4,
#     #     'sigma0': sigma0,
#     #     'run_name': run_name,
#     #     'env_type': env_type,
#     #     'game': game,
#     #     'r': rvalue,
#     #     'steps_max':steps_max,
#     # }

#     # algo = NCSCCAlgo(**kwargs)

def simulate(path,model):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()
    logger = Logger(path)
    with open('human_buf.pickle','rb+') as f:
        humanBuf = pickle.load(f)

    env = gym.make("PitfallNoFrameskip-v4")
    env = wrap_deepmind(env,frame_stack=4)

    policy = SupervisePolicy(env, network='Nature', nonlin_name='relu',humanBuf=humanBuf)
    vb = policy.get_vb()
    policy.set_vb(vb)

    param = logger.load_parameters(os.path.join(path,model))
    policy.set_parameters(param)
    #reward,length,seeds = [],[],[]
    save_action_list = []
    for i in range(1):
        seed = np.random.randint(100000)
        # seed = 94951
        rews, lens, action_list = policy.simulate(seed)
        #reward.append(rews)
        #length.append(lens)
        #seeds.append(seed)
        if rews > 0:
            with open(path+'/'+str(i)+'.pickle','wb+') as f:
                pickle.dump(rews,f)
                #obs_list = env.env.env.obs_list
                #clip = ImageSequenceClip(obs_list, fps=24)
                #clip.write_videofile(model+str(rank)+'.mp4', fps=24)
    return rews,lens,np.float(seed)

def simulate_1(path,model):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()
    logger = Logger(path)
    with open('human_buf.pickle','rb+') as f:
        humanBuf = pickle.load(f)

    env = gym.make("PitfallNoFrameskip-v4")
    env = wrap_deepmind(env,frame_stack=4)

    policy = SupervisePolicy(env, network='Nature', nonlin_name='relu',humanBuf=humanBuf)
    vb = policy.get_vb()
    policy.set_vb(vb)

    param = logger.load_parameters(os.path.join(path,model))
    policy.set_parameters(param)
    reward = []
    #reward,length,seeds = [],[],[]
    save_action_list = []
    for i in range(10):
        seed = np.random.randint(100000)
        rews, lens, action_list = policy.simulate(seed)
        reward.append(rews)
        if rews > 0:
            with open(path+'/'+str(i)+'.pickle','wb+') as f:
                pickle.dump(rews,f)
                #obs_list = env.env.env.obs_list
                #clip = ImageSequenceClip(obs_list, fps=24)
                #clip.write_videofile(model+str(rank)+'.mp4', fps=24)
    return np.mean(rews)

@click.command()
@click.option('-p', type=click.STRING)
@click.option('-m', type=click.STRING)
def main(p,m,ncpu=40):
    path = './logs_mpi/Pitfall/'+p
    start = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    def syncOneValue(v):
        v_t = np.array([v])
        v_all = np.zeros((cpus, 1))
        comm.Allgather([v_t, MPI.DOUBLE], [v_all, MPI.DOUBLE])
        return v_all.flatten()

    reward_list,frames_list,seeds_list = [],[],[]
    rew = simulate_1(path,m)
    reward = syncOneValue(rew)
    reward_list.extend(reward)
    if rank == 0:
        print(path)
        print("reward %s" % str(np.mean(reward)))
        print("time %s " %str(time.time()-start))
        with open(os.path.join(path,'test.txt'),'a') as f:
            f.write(path+'\n')
            f.write("reward %s  \n" % str(reward_list))
            f.write("mean reward %s  \n" % str(np.mean(reward_list)))
            #f.write("seed %s \n " % str(seeds_list))
main()

@click.command()
@click.option('--path', type=click.STRING)
def printDist(path):
    print(path[-20:])
    with open(path,'rb') as f:
        params = pickle.load(f)["parameters"]

    def calDist(params):
        max_ = np.max(params)
        min_ = np.min(params)
        mean = np.average(params)
        var = np.var(params)
        return max_,min_,mean,var

    print(calDist(params))


"""
run5-1 1216
(0.2531331, -0.24754675, -4.5903485e-06, 0.0024996307)
run5-2 1079
(0.24243768, -0.27524737, -9.3180864e-07, 0.0024959156)
run5 1200
(0.2338395, -0.23529446, 2.6999307e-06, 0.002493152)
run5-2 0
(0.24243768, -0.27524737, -9.3180864e-07, 0.0024959156)
/run1/parameters_564
(35.27310258863365, -33.14346694946289, 0.0069025999455193
015, 43.90509501130829)
8/run1/parameters_51
(11.088991795087304, -11.97671075085098, -0.00077890173201
0661, 3.679127423113297)
"""







