import numpy as np
import random
from matplotlib import pyplot


def distances_from_centres(X, centres):
    (M,N,C) = X.shape
    (K,C2) = centres.shape
    flag = True
    ret = None
    if C != C2:
        print("distances error")
    for c in centres:
        c = np.tile(c, (M*N, 1))
        distances = np.square(c - X.reshape((M*N, C)))
        distances = np.sum(distances, axis = 1)
        if flag:
            ret = distances
            flag = False
        else:
            ret = np.concatenate((ret,distances), axis=0)
    ret = ret.reshape((16, M*N)).T
    print(ret)
    return ret
def get_index(X, centres):
    (M,N,C) = np.shape(X)
    (K,C2) = np.shape(centres)
    if C != C2:
        print("assign index error")
    distances = distances_from_centres(X, centres)
    min_index = np.argmin(distances, axis = 1)
    return min_index

random.seed()
K = 16  #K
max_iteration = 10  #maximum number of iterations

img = pyplot.imread('in\lenna.jpg')
(M,N,dummy) = np.shape(img)
print(M)
print(N)
print(dummy)

num_pixel = M * N

#initialize 16 cluster centre
sample_index = np.array(random.sample(range(M*N), K))
cluster_centres = img.reshape(M*N, 3)[sample_index]
#initialize index array
index_arr = [0 for i in range(M*N)]

min_index = get_index(img, cluster_centres)

print(min_index)

