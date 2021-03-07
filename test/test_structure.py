import numpy as np
from mpi4py import MPI

# config and args dict
args = {
    "optimizer": "NCSOptimizer",
    "settings": {
    "learning_rate": 1,
    "sigma": 0.01,
    "c_sigma_factor": 1,
    "mu": 50
    },
    "network": "Nature",
    "nonlin_name": "relu",

    "group_cpus":3,
    "eva_times":1,
    "game":'Freeway',
    "run_name":'debug',
    "epoch": 5,
    "r":0.2,
    "timesteps_limit": 1e8,
    "D":1700000, # model size need to be set in run
    "d":10000,
    "m":10,
    "mu":3
}


class NCSRE(object):
	def __init__(self, args):
		self.comm = MPI.COMM_WORLD
		self.id = self.comm.Get_rank()
		self.cpus = self.comm.Get_size()
		self.group_cpus = args['group_cpus']
		self.group_id = int((self.id-1)/self.group_cpus)
		
        self.param = np.empty([10,10])
        self.D = len(self.param)
        self.d = 5
        self.eff_params = np.empty(self.d)
        self.eff_params_all = np.empty((self.cpus,self.d))

        # best score in subprocess
        self.BestScore = np.zeros(1) 
        self.Bestparams = np.empty(self.D)
        
        # line 10 , best params x of lam project y
        self.LocalBestparams = np.empty((self.group_cpus,self.D))
        self.LocalBesteffparams = np.empty((self.group_cpus,self.d))
        # line 12 , best params x of all local best
        self.GlobalBestparam = np.empty(self.D)
        self.GlobalBesteffparam = np.empty(self.d)
        self.LocalBestscores = np.zeros((self.group_cpus,1))
        self.GlobalBestscore = np.zeros(1)

    def communicate(self):
        if self.rank !=0:
            self.comm.send([self.Bestparams,MPI.FLOAT],[self.LocalBestparams[self.rank],MPI.FLOAT])

