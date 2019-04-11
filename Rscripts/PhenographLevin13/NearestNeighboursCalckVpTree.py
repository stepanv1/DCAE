#To evaluate Phenograph on uncurated data set
from sklearn.neighbors import KDTree
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import os
import sklearn
import fcsparser
import timeit
from multiprocessing import Pool
from contextlib import closing
from itertools import repeat
from numpy import linalg as LA

import numpy as np
import vptree
import tensorflow

def calc_jaccard(i, idx):
    """Compute the Jaccard coefficient between i and i's direct neighbors"""
    coefficients = np.fromiter((len(set(idx[i]).intersection(set(idx[j]))) for j in idx[i]), dtype=float)
    coefficients /= (2 * idx.shape[1] - coefficients)
    return idx[i], coefficients

def parallel_jaccard_kernel(idx):
    """Compute Jaccard coefficient between nearest-neighbor sets in parallel
    :param idx: n-by-k integer matrix of k-nearest neighbors
    :return (i, j, s): row indices, column indices, and nonzero values for a sparse adjacency matrix
    """
    n = len(idx)
    with closing(Pool()) as pool:
        jaccard_values = pool.starmap(calc_jaccard, zip(range(n), repeat(idx)))
    return jaccard_values

def euclidean(p1, p2):
  return np.sqrt(np.sum(np.power(p2 - p1, 2)))



SET_NAMES = ['Levine_32dim', 'Samusik_01', 'Samusik_all']
markers_dict = { 'Levine_32dim': range(4,36), 'Samusik_01': range(8,47), 'Samusik_all': range(8,47)}

for i in SET_NAMES:
#Set the variables
    ROOT = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL'
    DATA_DIR = "/benchmark_data_sets/"
    DATA = i + '.fcs'

    MANUAL_PHENOGRAPH = "/results/manual/phenoGraph/"
    RES_DIR_PHENOGRAPH = "/results/auto/PhenoGraph/"
    CALC_NAME = "kk30L2cosine"
    SET_NAME = i

    meta, data = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=False, reformat_meta=True, output_format='ndarray')
    meta['_channels_']

    #extract markers
    data = data[:,markers_dict[i]]


    #normalise to get cosine distance sqrt(2-xy)
    #data = sklearn.preprocessing.normalize(data, axis=1, norm='l2')


    #normalize data
    #data=   (data.T / LA.norm(data, ord=2, axis = 1)).T


    start = timeit.default_timer()
    tree = vptree.VPTree(data, euclidean)
    #nn = tree.get_n_nearest_neighbors(data, 31)
    data2 =data + np.random.normal(size=(161443,32))
    nList = [(0,0)]*161443
    for index, w in enumerate(data2):
        nList[index] = tree.get_n_nearest_neighbors(w, 31)

    ind += 1
    stop = timeit.default_timer()
    print(stop - start)

    if not os.path.exists(ROOT  + '/results/'+ CALC_NAME):
        os.makedirs(ROOT + '/results/'+CALC_NAME)
    np.savetxt(ROOT  + '/results/'+ CALC_NAME+"/"+"dist" + SET_NAME + ".txt", newline='\n',
               fmt='%f', X = dist)

    if not os.path.exists(ROOT  + '/results/'+ CALC_NAME):
        os.makedirs(ROOT  + '/results/'+ CALC_NAME)
    np.savetxt(ROOT  + '/results/'+ CALC_NAME+"/"+"ind" + SET_NAME + ".txt", newline='\n',
               fmt='%f', X = ind)



    ind = np.loadtxt(ROOT + '/results/' + CALC_NAME+ "/" + "ind" + SET_NAME + ".txt")
    dist = np.loadtxt(ROOT + '/results/' + CALC_NAME+ "/" + "dist" + SET_NAME + ".txt")
    ind = ind-1#jaccard distances

    pool = Pool(processes=10)
    jaccard30 = parallel_jaccard_kernel(ind[:,1:31])

    j30list = [list(y) for y in [zip([z]*30, jaccard30[z][0], jaccard30[z][1]) for z in [x for x in range(len(jaccard30))]]]

    dt=np.dtype('int, int, float')
    j30arrList = [[list(y) for y in x] for x in j30list]

    j30 = [np.array(x) for x in j30arrList]
    j30st = np.vstack(j30)

    np.savetxt(ROOT + '/results/'+ CALC_NAME+"/"+"j30" + SET_NAME + ".txt", newline='\n', fmt=['%d', '%d', '%f'], X = j30st)