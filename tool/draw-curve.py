import pickle
import matplotlib.pyplot as plt
import argparse

def args_parser():
    parser=argparse.ArgumentParser(description='Please input enviroment')
    
    parser.add_argument('-a','--algo',nargs='?',default='NCS',help='algorithm')
    parser.add_argument('-p','--path',nargs='?',default= None,help='path')
    parser.add_argument('-g','--game',nargs='?',default= None,help='game')
    parser.add_argument('-n','--name',nargs='?',default= 'rerun1',help='name')
    
    args=parser.parse_args()
    return args

def draw(algo,name,game,path):
    with open(path,'rb') as f:
        log = pickle.load(f)
    x = log['steps']
    y = log['performance']
    plt.plot(x, y)
    plt.title(algo+'-'+game+'-'+name)
    plt.ylabel('action')
    plt.xlabel('timesteps')
    plt.savefig(path.split('retest')[0]+algo+name+'.png')

args = args_parser()
draw(args.algo,args.name,args.game,args.path)
