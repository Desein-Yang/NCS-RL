import numpy as np
import time
def seed():
    seeds = []
    np.random.seed(int(time.time()))
    for i in range(4):
        seed = np.random.randint(999999)
        for j in range(3):
            seeds.append(seed)
    return seeds


def getRandom(ini_seed,j):
    D = 10
    d = 5
    rng = np.random.RandomState(ini_seed)
    child_seed = rng.randint(0,999999,size = (D,))
    rng = np.random.RandomState(child_seed[j])
    random_vector = rng.standard_normal((5,d))
    #print(random_vector)
    #print(child_seed)
    return random_vector

# x = np.empty((10,))
# # r = getRandom(1000,5)
# # ans = np.dot(r , np.ones((1000,)))
# for j in range(0,10,5):
#     x[j:j+5] = np.dot(getRandom(1234,j), np.ones((5,)))

# print(x)
np.random.seed(1)
for i in range(3):
    print(np.random.uniform(0,1,(2,1)))
