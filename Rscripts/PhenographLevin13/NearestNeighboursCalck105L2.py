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


SET_NAMES = ['Levine_13dim', 'Levine_32dim', 'Samusik_01', 'Samusik_all']
markers_dict = {'Levine_13dim': range(0,13) , 'Levine_32dim': range(4,36), 'Samusik_01': range(8,47), 'Samusik_all': range(8,47)}

for i in SET_NAMES:
#Set the variables
    ROOT = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL'
    DATA_DIR = "/benchmark_data_sets/"
    DATA = i + '.fcs'

    MANUAL_PHENOGRAPH = "/results/manual/phenoGraph/"
    RES_DIR_PHENOGRAPH = "/results/auto/PhenoGraph/"
    CALC_NAME = "kk105L2"
    SET_NAME = i

    meta, data = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=False, reformat_meta=True, output_format='ndarray')
    meta['_channels_']

    #extract markers
    data = data[:,markers_dict[i]]






    #subset the data


    start = timeit.default_timer()
    tree = KDTree(data, metric='euclidean')
    dist, ind = tree.query(data, k=106, dualtree = True)
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
    jaccard30 = parallel_jaccard_kernel(ind[:,1:106])

    j30list = [list(y) for y in [zip([z]*105, jaccard30[z][0], jaccard30[z][1]) for z in [x for x in range(len(jaccard30))]]]

    dt=np.dtype('int, int, float')
    j30arrList = [[list(y) for y in x] for x in j30list]

    j30 = [np.array(x) for x in j30arrList]
    j30st = np.vstack(j30)

    np.savetxt(ROOT + '/results/'+ CALC_NAME+"/"+"j105" + SET_NAME + ".txt", newline='\n', fmt=['%d', '%d', '%f'], X = j30st)