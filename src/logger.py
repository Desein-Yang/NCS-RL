import pickle
import os
import numpy as np
import time
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_name = "log.txt"
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except FileExistsError:
                pass

    def log(self, message):
        with open(os.path.join(self.log_dir, self.log_name), "a") as f:
            f.write(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))+" "+message+"\n")

    def set_logdir(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except FileExistsError:
                pass
    
    def set_logname(self, name):
        self.log_name = name

    def draw_single(self,log_dict,game,algo,savename='train-curve',save=True):
        """draw train curve"""
        assert 'steps' in log_dict.keys(),"log error"
        assert 'performance' in log_dict.keys(),"log error"
        x = log_dict['steps']
        y = log_dict['performance']
        plt.plot(x, y)
        plt.title(algo + '-' + game)
        plt.ylabel('score')
        plt.xlabel('timesteps')
        if save:
            plt.savefig( os.path.join(self.log_dir,savename+'.png'))
        else:
            plt.show()

    def draw_two(self,log1,log2,game,algo,savename="train-test-curve",save=True):
        assert 'steps' in log1.keys(),"log error"
        assert 'performance' in log1.keys(),"log error"
        assert 'steps' in log2.keys(),"log error"
        assert 'performance' in log2.keys(),"log error"
        x1 = log1['steps']
        y1 = log1['performance']
        x2 = log2['steps']
        y2 = log2['performance']
        plt.plot(x1, y1, label='test')
        plt.plot(x2, y2,label = 'train')
        plt.title(algo + '-' + game)
        plt.legend()
        plt.ylabel('score')
        plt.xlabel('timesteps')
        if save:
            plt.savefig( os.path.join(self.log_dir,savename+'.png'))
        else:
            plt.show()

    def write_general_stat(self, stat_string):
        with open(os.path.join(self.log_dir, "stat.txt"), "a") as f:
            f.write(stat_string)

    def write_optimizer_stat(self, stat_string):
        if stat_string is not None:
            with open(os.path.join(self.log_dir, "optimizer_stat.txt"), "a") as f:
                f.write(stat_string)

    def save_parameters(self, parameters, iteration):
        with open(os.path.join(self.log_dir, "parameters_%d" % iteration), 'wb') as f:
            pickle.dump({"parameters": parameters}, f)

    def save_vb(self, vb):
        np.save(os.path.join(self.log_dir, "vb.npy"), vb)

    def log_for_debug(self, message):
        with open(os.path.join(self.log_dir, "debug_log.txt"), "a") as f:      
            t = time.localtime(time.time())
            m = (str(t.tm_mon) + "-" + str(t.tm_mday) + " " + str(t.tm_hour) + ":" + str(t.tm_min) +":"+str(t.tm_sec) + message)
            f.write( m +"\n")
        
    def load_parameters(self,path):
        with open(path, 'rb') as f:
            params = pickle.load(f)["parameters"]
        return params
