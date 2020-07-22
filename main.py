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
    ret = ret.reshape((K, M*N)).T
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
def new_cluster_centres(X, min_index, K):
    (M, N, C) = np.shape(X)
    total = np.zeros((K, C))
    total_samples = np.zeros(K)
    for i in range(M*N):
        total[min_index[i]] += X.reshape((M*N, C))[i]
        total_samples[min_index[i]] += 1
    total_samples = np.atleast_2d(total_samples).T
    total_samples = np.tile(total_samples, (1,C))
    avg_distance = total / total_samples
    return avg_distance
def rebuild_image(size, min_index, cluster_centres):
    (M,N,C) = size
    new_image = np.zeros((M*N,C))
    cc = np.around(cluster_centres)
    for i in range(M*N):
        new_image[i] = cc[min_index[i]]
    new_image = new_image.reshape((M,N,C))
    return new_image

def compress_img(img, K, max_iteration):
    random.seed()
    (M,N,C) = np.shape(img)
    num_pixel = M * N

    #initialize cluster centre
    sample_index = np.array(random.sample(range(M*N), K))
    cluster_centres = img.reshape(M*N, 3)[sample_index]
    #initialize index array
    min_index = None
    for iteration in range(max_iteration):
        #print(str(iteration + 1) + "\t/" + str(max_iteration))
        #compute the average of each group of samples
        min_index = get_index(img, cluster_centres)
        #update the cluster centres with the average value of assigned samples
        cluster_centres = new_cluster_centres(img, min_index, K)
    new_img = rebuild_image((M,N,C), min_index, cluster_centres)
    return new_img.astype(np.int32)

img = pyplot.imread('bird.jpg')
compressed_img16 = compress_img(img, 16, 10)
fig = pyplot.figure()
a = fig.add_subplot(1, 2, 1)
imgplot = pyplot.imshow(img)
a.set_title("Original")

b = fig.add_subplot(1, 2, 2)
imgplot = pyplot.imshow(compressed_img16)
b.set_title("K = 16")

pyplot.show()