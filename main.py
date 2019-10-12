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
        return
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
    return ret
def get_index(X, centres):
    (M,N,C) = np.shape(X)
    (K,C2) = np.shape(centres)
    if C != C2:
        print("assign index error")
        return
    distances = distances_from_centres(X, centres)
    min_index = np.argmin(distances, axis = 1)
    return min_index

random.seed()
K = 16  #K
max_iteration = 10  #maximum number of iterations

img = pyplot.imread('in\lenna.jpg')
(M,N,C) = np.shape(img)

num_pixel = M * N

#initialize 16 cluster centre
sample_index = np.array(random.sample(range(M*N), K))
cluster_centres = img.reshape(M*N, 3)[sample_index]
#initialize index array
min_index = None
for iteration in range(max_iteration):
    #compute the average of each group of samples
    min_index = get_index(img, cluster_centres)
    total = np.zeros((16,3))
    total_samples = np.zeros(16)
    for i in range(M*N):
        total[min_index[i]] += img.reshape((M*N, C))[i]
        total_samples[min_index[i]] += 1
    total_samples = np.atleast_2d(total_samples).T
    total_samples = np.tile(total_samples, (1,3))
    avg_distance = total / total_samples
    #update the cluster centres with the average value of assigned samples
    cluster_centres = avg_distance
cluster_centres = np.around(cluster_centres, decimals = 0)
compressed_img = np.zeros((M*N,C))
for i in range(M*N):
    compressed_img[i] = cluster_centres[min_index[i]]
compressed_img = compressed_img.reshape((M,N,C))
compressed_img = np.uint8(compressed_img)
print(compressed_img)
imgplot = pyplot.imshow(compressed_img)
pyplot.show()
