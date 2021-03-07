from analysis import EnvAnalysis
import gym
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game','-g',default=None)
    parser.add_argument('--runname','-r',default='debug')
    args = parser.parse_args()
    game = args.game
    runname = args.runname
    try:
        # if game in os.listdir('./logs_randseed'):
        #    raise Exception
        # else:
        #     
        env = gym.make("%sNoFrameskip-v4" % game)
    except:
        pass
    else:
        exp = EnvAnalysis(game,env)
        exp.randomExp(runname)
    # env = gym.make("%sNoFrameskip-v4" % game)

# print(exp.rollout(seed=12346)

# ([185.0, 70.0, 185.0, 100.0, 70.0, 100.0, 70.0, 100.0, 185.0, 185.0, 70.0, 100.0, 100.0, 70.0, 185.0, 100.0, 70.0, 70.0, 185.0, 185.0, 70.0, 185.0, 70.0, 70.0, 185.0, 100.0, 100.0, 70.0, 70.0, 100.0], [968, 525, 967, 399, 528, 398, 531, 403, 966, 965, 525, 402, 396, 531, 962, 402, 528, 525, 961, 970, 527, 970, 524, 530, 967, 402, 395, 524, 530, 397], 100.0, 397.0)
# 同一个随机种子不同的随机帧居然跑的不一样？？