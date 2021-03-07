import numpy as np

def test(a):
    assert len(a.shape) == 2
    a[0][0] = 100
    a[a>1] = 1000
    print("in change:", a)
    print("in change param id:", id(a))

if __name__ == '__main__':
    data = np.ones((2, 2))
    print("before change: ", data)
    print("param id:", id(data))
    test(data)

    print("after change: ", data)
    print("after change param id:", id(data))
