import numpy as np

class Decomposer(object):

    def __init__(self, group, randomSeed):
        # the number of subcomponents
        self.condidate_size = group
        self.randomState = np.random.RandomState(randomSeed)

    def get_mask(self, n):
        '''
        return:
            m: 参数分组的数量
            mask: 大小为n的向量，值为[0,1,2,..., m-1]
        '''
        m = self.randomState.choice(self.condidate_size)
        mask = self.randomState.uniform(0, 1, n)
        for i in range(m):
            mask[(mask>=i/m) & (mask<(i+1)/m)] = i
        return m, mask