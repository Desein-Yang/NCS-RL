import numpy as np

# 暂时只支持f1-f6
def f1(x, o, bias):
    return np.sum((x-o)*(x-o)) + bias    

def f2(x, o, bias):
    x_o = np.abs(x - o)
    return np.max(x_o) + bias

def f3(x, o, bias):
    z = x - o + 1
    tmp = z[:-1] * z[:-1] - z[1:]
    part1 = 100 * np.sum(tmp * tmp)
    part2 = np.sum(((z-1)*(z-1))[:-1])
    return part1 + part2 + bias 

def f4(x, o, bias):
    z = x - o
    return np.sum(z*z - 10*np.cos(2*np.pi*z)+10)+bias

def f5(x, o, bias):
    z = x - o
    part1 = np.sum(z*z/4000)
    n = x.shape[0]
    I = np.arange(1, n+1, 1)
    tmp = np.cos(z/np.sqrt(I))
    part2 = -tmp.prod()
    return part1 + part2 + bias + 1

def f6(x, o, bias):
    z = x - o
    n = x.shape[0]
    return -20*np.exp(-0.2*np.sqrt(1/n*np.sum(z*z))) - np.exp(1/n*np.sum(np.cos(2*np.pi*z)))+20+np.e+bias

def f7(x, o, bias):
    pass

test_func_dict = {
            1:f1,
            2:f2, 
            3:f3,
            4:f4,
            5:f5,
            6:f6,
            7:f7
        }

test_func_bound = {
    1:[-100, 100],
    2:[-100, 100],
    3:[-100, 100],
    4:[-5, 5],
    5:[-600, 600],
    6:[-32, 32],
    7:[0, 0],
}

test_func_bias = [0,  -450.0 , -450.0 , 390.0 , -330.0 , -180.0 , -140.0]


class TestEnv(object):

    def __init__(self, d, problem_t):
        self.d = d
        self.problem_t = problem_t
        if problem_t != 7:
            self.o = np.load("data/f%d.npy" % problem_t)
            self.o = self.o[:d]
            # self.bias = test_func_bias[problem_t]
            self.bias = 0
        else:
            raise ValueError("wrong problem type")

    def get_fitness(self, x):
        return test_func_dict[self.problem_t](x, self.o, self.bias)

