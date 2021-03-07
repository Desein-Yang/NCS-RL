#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_bound.py
@Time    :   2020/05/02 09:04:28
@Author  :   Qi Yang
@Version :   1.0
@Describtion:  测试边界范围
'''

# here put the import lib
import numpy as np
from numpy.random import default_rng
import cProfile
import time

def getRandomMatrix(seed,D,d):
    """generate a pesudo random matrix with seed.  
    Return:   
        a pesudo matrix [D,d] related with seed.  
    """
    rng = np.random.RandomState(seed)
    A = rng.standard_normal((D, d),dtype=np.double)
    return A

def getRandomVector2(ini_seed,idx,s,D,d):
    """generate a [s,d] random vector seed generated with ini_seed.  
    ini_seed is grandpa seed."""
    rng = np.random.RandomState(ini_seed)
    child_seed = rng.randint(0,999999,size = (D,))
    rng = np.random.RandomState(child_seed[idx])
    random_vector = rng.standard_normal((s,d))
    return random_vector

def getRandomVector(ini_seed,idx,s,D,d):
    rng = default_rng(ini_seed)
    child_seed = rng.integers(0,999999,size = (D,))
    rng2 = default_rng(child_seed[idx])
    return rng2.standard_normal((s,d))


def from_y_to_x(x,y,seeds,idx,s,D,d):
    """map params y in d-dimension to x D-dimension
    v1: x = A * y  
    v2: x' = ax + A*y
    v3: x'[j] = ax + A[j]*y 
    """
    tmp = np.empty((D,))
    if idx is not 0:
        for j in range(0,D,s):
            if j+s > D:
                s = D-1-j
            tmp[j:j+s,] = np.dot(getRandomVector(seeds[idx],j,s,D,d),y)
        x = 1.0 * x + tmp
    else:
        x = tmp
    return x

def getSeed(mu,lam):
    """get seeds list and seeds in a group is same.   
    E.g.[232913,232913,232913,345676,345676,345676,894356,894356,894356] 
    """
    seeds = [0]
    np.random.seed(int(time.time()))
    for i in range(mu):
        seed = np.random.randint(999999)
        for j in range(lam):
            seeds.append(seed)
    return np.array(seeds,dtype = np.int64)

def test_vector(s,D,d):
    seeds = getSeed(3,3)
    y = np.ones((d,))
    x = np.zeros((D,))
    x = from_y_to_x(x,y,seeds,3,s,D,d)
    print(x[0:10,])

def test_bound(D,d,l,h):
    seeds = getSeed(3,3)
    y = np.clip(np.random.rand(d,)*2-1,l,h)
    x = np.zeros((D,))
    print(calDist(y))
    x = from_y_to_x(x,y,seeds,3,100000,D,d)
    print(calDist(x))

def multi_test_bound(D,d,l,h,g):
    y = np.clip(np.random.rand(d,)*2-1,l,h)
    for i in range(g):
        seeds = getSeed(3,3)
        x = np.zeros((D,))
        print('g'+str(i))
        print(calDist(y))
        x = from_y_to_x(x,y,seeds,3,100000,D,d)
        print('g'+str(i))
        print(calDist(x))
        y += np.random.randn(d,)
        y = np.clip(y,l,h)
    return x

def test_random_vector():
    ss = [5000,10000,50000,100000,150000]
    for i in range(5):
        y,t = getRandomVector(123456,3,ss[i],1681230,10000)
        print(t)

def test_random_vector2():
    for i in range(3):
        y = getRandomVector(123456,3,1,100,10)
        print(y)


def calDist(params):
    mean_ = np.around(np.mean(params),decimals=2)
    min_ = np.around(np.min(params),decimals=2)
    max_ = np.around(np.max(params),decimals=2)
    sigma = np.around(np.var(params),decimals=2)
    return mean_,min_,max_,sigma
# 代码里
if __name__ == "__main__":

    # test_bound(1600000,100,-5,5)
    # test_bound(1600000,100,-1,1)
    # test_bound(1600000,100,-0.5,0.5)
    # test_bound(1600000,100,-0.1,0.1)
    
    # test_bound(1600000,100,-0.01,0.01)

    # multi_test_bound(1600000,100,-0.05,0.05,10)
    test_bound(1600000,100,-0.1,0.1)

'''
测试了y的范围：
(0.03, -0.99, 0.99, 0.39)
(-0.01, -34.85, 30.76, 39.2)

(-0.09, -1.0, 0.99, 0.31)
(0.0, -28.09, 27.23, 31.4)

(-0.01, -0.5, 0.5, 0.16)
(0.0, -18.91, 20.44, 16.13)

(-0.01, -0.1, 0.1, 0.01)
(0.0, -4.97, 5.07, 0.94)

(-0.0, -0.01, 0.01, 0.0)
(0.0, -0.51, 0.47, 0.01)

所以选定搜索范围为 x [-5,5],y [-1,1]
'''
