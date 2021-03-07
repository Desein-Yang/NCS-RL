import gym
import atari_py as ap
from gym.spaces import Discrete, Box

def getGamelist():
    game_list = ap.list_games()
    g = []
    for game in sorted(game_list):
        if '_' in game or game[0].islower():
            game = "".join(map(lambda x:x.capitalize(), game.split("_")))
        g.append(game)
    return g

def getType(space):
    if isinstance(space,Discrete):
        t = 'Discrete'
    elif isinstance(space,Box):
        t = 'Box'
    else:
        t = 'other'
    return t

def analysis_atari():
    game_list = ap.list_games()
    # with open('game_list.txt','a') as f:
    #     for i in sorted(game_list):
    #         f.write(str(i)+'\n')
    with open('env_record.csv','a') as f:
        f.write('env_id,Game/Task,ac_space,actions,ob_space,ob_shapes,render_mode1,render_mode2,max_episode_sec,max_episode_steps,elapsed_steps,episode_started_at\n')
    for game in sorted(game_list):
        if '_' in game or game[0].islower():
            game = "".join(map(lambda x:x.capitalize(), game.split("_")))
        try:
            env = gym.make("%sNoFrameskip-v4" % str(game))
            ac = env.action_space
            actype = getType(ac)
            acn = ac.n
            reward_range = env.reward_range
            ob = env.observation_space
            obtype = getType(ob)
            obshape = ob.shape

            attr_dict = env.__dict__
            render_mode_1 = env.metadata['render.modes'][0]
            render_mode_2 = env.metadata['render.modes'][1]
            env_id = attr_dict['_env_closer_id']
            max_episode_sec = attr_dict['_max_episode_seconds']
            max_episode_steps = attr_dict['_max_episode_steps']
            elapsed_steps = attr_dict['_elapsed_steps'] 
            episode_started_at = attr_dict['_episode_started_at']

            l = [str(env_id),game,actype,str(acn),obtype,str(obshape),render_mode_1,render_mode_2,str(max_episode_sec),str(max_episode_steps),str(elapsed_steps),str(episode_started_at),'\n']
            delimiter = ','
            with open('env_record.csv','a') as f:
                f.write(delimiter.join(l))
        except:
            pass
        
g = getGamelist()
print(g)


