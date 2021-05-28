import sys
sys.path.insert(0, '/home/sgrinek/PycharmProjects/vpsearch_python_binding')
import vpsearch as vp
import numpy as np
import falconn
import timeit

data =  np.random.rand(100000, 40)

number_of_tables = 50

params_cp = falconn.LSHConstructionParameters()
params_cp.dimension = len(data[0])
params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
params_cp.l = number_of_tables
params_cp.num_rotations = 1
params_cp.num_setup_threads = 0
table = falconn.LSHIndex(params_cp)
params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
falconn.compute_number_of_hash_functions(18, params_cp)
print('Constructing the LSH table')
t1 = timeit.default_timer()
table = falconn.LSHIndex(params_cp)
table.setup(data)
t2 = timeit.default_timer()
print('Done')
print('Construction time: {}'.format(t2 - t1))

query_object = table.construct_query_object()


    # final evaluation
t1 = timeit.default_timer()
zzz=query_object.find_k_nearest_neighbors(data[0,:], 31)
t2 = timeit.default_timer()

print('Query time: {}'.format((t2 - t1) / 1))









indVP, distVP  = vp.find_nearest_neighbors(data, 30, 1)

from  sklearn.neighbors import BallTree
tree = BallTree(data, leaf_size=40,  metric="euclidean")
dist, ind = tree.query(data[:100,:], k=31)

np.sum(ind[:,1:31]!=indVP[:100,:])
#number of errorrs grows with the number of neighbor


print(np.where(ind[:,1:31]!=indVP[:100,:]))





from libKMCUDA import kmeans_cuda, knn_cuda
ca = kmeans_cuda(np.float32(data), 25, metric="euclidean", verbosity=1, seed=3, device=0)
neighbors = knn_cuda(30, np.float32(data), *ca, metric="euclidean", verbosity=1, device=0)
print(neighbors[0])
sum(ind[:,1:31]!=neighbors[:100,:])