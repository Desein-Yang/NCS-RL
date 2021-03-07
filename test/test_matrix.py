import numpy as np
import cProfile

class matrix(object):
    def __init__(self,d,D):
        self.d = d
        self.D = D
        
        self.random_seed = np.random.randint(100000)
        self.randomstate = np.random.RandomState(self.random_seed)
        self.A = self.random_matrix()

    def run(self):
        self.x = np.ones(self.D)
        self.y = self.random_embedding(self.x)
        print(self.y)
        print(x)
        
        # Returns False because the first key is false.
        # For dictionaries the all() function checks the keys, not the values.
    def random_matrix(self):
        '''generate A, line 4.has tested.    
        Args:  
          mean and sigma of Gaussian
          D,d
        Return:  
          Matrix A of [D,d]
        '''
        return self.randomstate.standard_normal((self.d,self.D))

    def random_embedding(self,x):
        
        print(self.A)
        print(self.x)
        #y1 = np.dot(np.linalg.tensorinv(self.A),x)
        #y2 = np.linalg.tensorsolve(self.A,x)
        #y2 = np.linalg.tensorsolve(self.A,x)
        y3 = np.dot(np.linalg.pinv(self.A),x)
        return y3 # y


# if __name__ == "__main__":
#     test = matrix(2,5)
#     test.run()
    # cProfile.run("test.run()")
d =10
y = np.empty((d,))
print(y.shape[0])
