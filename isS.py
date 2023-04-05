import numpy as np

def isS(p):
    ll = linefun(p)
    flag = np.sum(np.sum(ll * ll)) != np.sum(np.sum(np.abs(ll * ll)))
    return flag


def linefun(p):
    num = p.shape[0]
    ll = np.zeros(num - 2)
    for i in range(num - 2):
        ll[i] = (p[i, 1] - p[num - 1, 1]) / (p[0, 1] - p[num - 1, 1]) - (p[i, 0] - p[num - 1, 0]) / (
                    p[0, 0] - p[num - 1, 0])
    return ll
