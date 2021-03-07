#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_time.py
@Time    :   2020/05/02 09:04:28
@Author  :   Qi Yang
@Version :   1.0
@Describtion:  测试random embedding 时间怎么分最高效
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

from test_noisetable_mpi import NoiseTable
sharedNoisetable = NoiseTable(1000,123)
def getRandomVector3(size,idx=None):
    """Generate a [s,d] random vector From Noise Table"""
    assert type(size) is tuple
   
    
    if idx is None:
        idx = np.random.randint()
    return sharedNoisetable.randn(idx,size)

def from_y_to_x1(x,y,seeds,idx,s,D,d):
    """Map effective params y (d-dimension) to x (D-dimension).   
        v1: x = A * y   
        v2: x' = ax + A*y 
        v3: x'[j] = ax + A[j]*y  
    """
    tmp = np.empty((D,))
    assert s <= D
    if idx is not 0:
        iniseed = seeds[idx]
        idx_l = np.random.RandomState(iniseed).randint(low=1,size=(len(range(0,D,s)),))
        # import pdb; pdb.set_trace()
        for i,j in enumerate(range(0,D,s)):
            if j + s > D:
                s = D - 1 - j
            a =  getRandomVector3(
                    size = (s,d), 
                    idx = idx_l[i])
            # import pdb; pdb.set_trace()
            tmp[j:j+s,] = np.dot(a,y)
        x = 1.0 * x + tmp
    else:
        x = tmp
    return x

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
    for i in range(20):
        x = from_y_to_x1(x,y,seeds,3,s,D,d)
    print(x[0:10,])
    x = from_y_to_x(x,y,seeds,3,s,D,d)
    print(x[0:10,])

def test_random_vector():
    ss = [5000,10000,50000,100000,150000]
    for i in range(5):
        y,t = getRandomVector(123456,3,ss[i],1681230,10000)
        print(t)

def test_random_vector2():
    for i in range(3):
        y = getRandomVector(123456,3,1,100,10)
        print(y)



# 代码里
if __name__ == "__main__":
    print('vector time')
    # test_vector(10,100,10)
    cProfile.run("test_vector(10,100,10)",filename='vector_time.out')
    # cProfile.run("test_vector(1000,1681230,10000)",filename="vector_time.out")
    # cProfile.run("test_random_vector()",filename="vector_time.out")
    # test_random_vector()

    # 30000 310s
    # 100000 311s
    # 200000 313s
    # 300000 316s
    import pstats
    # 创建Stats对象
    p = pstats.Stats("vector_time.out")

    # strip_dirs(): 去掉无关的路径信息
    # sort_stats(): 排序，支持的方式和上述的一致
    # print_stats(): 打印分析结果，可以指定打印前几行

    # 和直接运行cProfile.run("test()")的结果是一样的
    #p.strip_dirs().sort_stats(-1).print_stats()

    # 按照函数名排序，只打印前3行函数的信息, 参数还可为小数,表示前百分之几的函数信息 
    p.strip_dirs().sort_stats("time").print_stats(0.5)

    # 按照运行时间和函数名进行排序
    # p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.5)

    # 如果想知道有哪些函数调用了sum_num
    # p.print_callers(0.5, "sum_num")

    # 查看test()函数中调用了哪些函数
    # p.print_callees("test")
'''
主要时间都花在获取随机向量上；
特别是 standard normal
dot不花时间；
'''
'''
Sat May  2 09:45:40 2020    vector_time.out 
D = 10000
d = 100
slice_gap = 1

         680049 function calls in 7.511 seconds

   Ordered by: cumulative time, function name
   List reduced from 48 to 24 due to restriction <0.5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    7.511    7.511 {built-in method builtins.exec}
        1    0.000    0.000    7.511    7.511 <string>:1(<module>)
        1    0.000    0.000    7.511    7.511 test_time.py:61(test_vector)
        1    0.050    0.050    7.511    7.511 test_time.py:34(from_y_to_x)
    10000    2.684    0.000    7.402    0.001 test_time.py:25(getRandomVector)
    20000    2.665    0.000    2.998    0.000 contextlib.py:49(inner)
    10003    0.870    0.000    1.243    0.000 {method 'randint' of 'numpy.random.mtrand.RandomState' objects}
    40001    0.137    0.000    0.368    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
    40000    0.111    0.000    0.262    0.000 _ufunc_config.py:39(seterr)
    10000    0.014    0.000    0.260    0.000 <__array_function__ internals>:2(prod)
    10000    0.024    0.000    0.231    0.000 fromnumeric.py:2792(prod)
    10000    0.047    0.000    0.207    0.000 fromnumeric.py:73(_wrapreduction)
    20000    0.036    0.000    0.191    0.000 _ufunc_config.py:441(__enter__)
    20000    0.055    0.000    0.144    0.000 random.py:680(getrandbits)
    20000    0.030    0.000    0.137    0.000 _ufunc_config.py:446(__exit__)
    10000    0.134    0.000    0.134    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    10000    0.119    0.000    0.119    0.000 {method 'standard_normal' of 'numpy.random.mtrand.RandomState' objects}
    20000    0.027    0.000    0.118    0.000 <__array_function__ internals>:2(concatenate)
    10003    0.040    0.000    0.112    0.000 _dtype.py:319(_name_get)
    40000    0.090    0.000    0.097    0.000 _ufunc_config.py:139(geterr)
    20000    0.060    0.000    0.097    0.000 abc.py:180(__instancecheck__)
    10003    0.021    0.000    0.073    0.000 numerictypes.py:365(issubdtype)
    20000    0.069    0.000    0.069    0.000 {built-in method posix.urandom}
    10000    0.014    0.000    0.059    0.000 <__array_function__ internals>:2(dot)
'''

'''
Sat May  2 09:55:28 2020    vector_time.out
D = 10000
d = 100
s = 10
         68049 function calls in 0.905 seconds

   Ordered by: cumulative time, function name
   List reduced from 48 to 24 due to restriction <0.5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.905    0.905 {built-in method builtins.exec}
        1    0.000    0.000    0.905    0.905 <string>:1(<module>)
        1    0.000    0.000    0.905    0.905 test_time.py:61(test_vector)
        1    0.009    0.009    0.905    0.905 test_time.py:34(from_y_to_x)
     1000    0.299    0.000    0.888    0.001 test_time.py:25(getRandomVector)
     2000    0.297    0.000    0.336    0.000 contextlib.py:49(inner)
     1003    0.097    0.000    0.139    0.000 {method 'randint' of 'numpy.random.mtrand.RandomState' objects}
     1000    0.074    0.000    0.074    0.000 {method 'standard_normal' of 'numpy.random.mtrand.RandomState' objects}
     4001    0.016    0.000    0.042    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
     4000    0.013    0.000    0.031    0.000 _ufunc_config.py:39(seterr)
     1000    0.002    0.000    0.029    0.000 <__array_function__ internals>:2(prod)
     1000    0.003    0.000    0.026    0.000 fromnumeric.py:2792(prod)
     1000    0.005    0.000    0.023    0.000 fromnumeric.py:73(_wrapreduction)
     2000    0.005    0.000    0.023    0.000 _ufunc_config.py:441(__enter__)
     2000    0.003    0.000    0.016    0.000 _ufunc_config.py:446(__exit__)
     2000    0.006    0.000    0.016    0.000 random.py:680(getrandbits)
     1000    0.015    0.000    0.015    0.000 {method 'reduce' of 'numpy.ufunc' objects}
     1003    0.004    0.000    0.013    0.000 _dtype.py:319(_name_get)
     2000    0.003    0.000    0.013    0.000 <__array_function__ internals>:2(concatenate)
     4000    0.010    0.000    0.011    0.000 _ufunc_config.py:139(geterr)
     2000    0.006    0.000    0.010    0.000 abc.py:180(__instancecheck__)
     1003    0.002    0.000    0.009    0.000 numerictypes.py:365(issubdtype)
     1000    0.002    0.000    0.008    0.000 <__array_function__ internals>:2(dot)
     2000    0.007    0.000    0.007    0.000 {built-in method posix.urandom}
'''

'''
vector time
D = 10000
d = 100
s = 100
Sat May  2 09:56:33 2020    vector_time.out

         6849 function calls in 0.174 seconds

   Ordered by: cumulative time, function name
   List reduced from 48 to 24 due to restriction <0.5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.174    0.174 {built-in method builtins.exec}
        1    0.000    0.000    0.174    0.174 <string>:1(<module>)
        1    0.000    0.000    0.174    0.174 test_time.py:61(test_vector)
        1    0.001    0.001    0.173    0.173 test_time.py:34(from_y_to_x)
      100    0.035    0.000    0.170    0.002 test_time.py:25(getRandomVector)
      100    0.077    0.001    0.077    0.001 {method 'standard_normal' of 'numpy.random.mtrand.RandomState' objects}
      200    0.034    0.000    0.038    0.000 contextlib.py:49(inner)
      103    0.012    0.000    0.017    0.000 {method 'randint' of 'numpy.random.mtrand.RandomState' objects}
      401    0.002    0.000    0.006    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
      100    0.000    0.000    0.003    0.000 <__array_function__ internals>:2(prod)
      400    0.001    0.000    0.003    0.000 _ufunc_config.py:39(seterr)
      100    0.000    0.000    0.003    0.000 fromnumeric.py:2792(prod)
      100    0.001    0.000    0.003    0.000 fromnumeric.py:73(_wrapreduction)
      200    0.000    0.000    0.002    0.000 _ufunc_config.py:441(__enter__)
      200    0.001    0.000    0.002    0.000 random.py:680(getrandbits)
      100    0.002    0.000    0.002    0.000 {method 'reduce' of 'numpy.ufunc' objects}
      200    0.000    0.000    0.002    0.000 _ufunc_config.py:446(__exit__)
      100    0.000    0.000    0.002    0.000 <__array_function__ internals>:2(dot)
      103    0.001    0.000    0.002    0.000 _dtype.py:319(_name_get)
      200    0.000    0.000    0.001    0.000 <__array_function__ internals>:2(concatenate)
      400    0.001    0.000    0.001    0.000 _ufunc_config.py:139(geterr)
      200    0.001    0.000    0.001    0.000 abc.py:180(__instancecheck__)
      103    0.000    0.000    0.001    0.000 numerictypes.py:365(issubdtype)
      200    0.001    0.000    0.001    0.000 {built-in method posix.urandom}
'''
