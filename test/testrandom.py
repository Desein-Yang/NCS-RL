#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   testrandom.py
@Time    :   2020/09/04 17:24:36
@Author  :   Qi Yang
@Version :   1.0
@Describtion:  Use for test random seed pool generation
'''
import numpy as np 


def get_seed_pool(n_train,n_test,seed=None,zero_shot=True,range=1e7):
    """Create train and test random seed pool.  
    If use zero shot performance, test seed will non-repeatitive.  """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.default_rng()
    Allseed = np.arange(range)
    trainset = rng.choice(Allseed,size=n_train,replace=False)  # means sample without replacement
    tmp = np.setdiff1d(Allseed,trainset,assume_unique=True) # get disjoint set of trainset
    if zero_shot is False:
        testset = rng.choice(tmp,size=n_test,replace=False)
    else:
        testset = rng.choice(Allseed,size=n_test,replace=False)
    return trainset, testset

import time
a = time.time()
trainset, testset = get_seed_pool(10000,10000,range=1e6)
print(trainset,testset)
print(time.time()-a)
