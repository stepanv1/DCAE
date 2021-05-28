
#TODO: realize global structure measure
# General idea for paper: on tyeh sphere there can be N fartherst points,
# while inside circle ~sqrt(M) - -beter preservation of topology


def k_farthest_neib_sphere():
    '''
    build knn tree (kd or ball)
    find point opposite to given point and
    query tree for its knn, they will be fartherst points on the sphere

    '''
    pass
def k_farthest_neib_plane():
    '''
    similar idea to farthestspoints_sphere
    instead of convex hull find knn and closest to
    the opposite point on the eclosing circle will be the fartherst one
    or just get the hull
    :return:
    '''
    pass

data, color = datasets.make_classification(n_samples=200000, n_features=40,  n_informative=10, n_redundant=0, n_repeated=0,
                n_classes=10, n_clusters_per_class=1, weights=None, flip_y=0.5, class_sep=100.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=12345)

#https://www.ratml.org/pub/pdf/2017exploiting.pdf
query = data[:,:]
data.shape
nb = find_neighbors(data = data, k_=90, metric = 'euclidean')

def k_farthest_neib_HD_mlpack(query, data, knnIdx, k, exact = True, percentage=1.0):
    '''
    using mlpack to find furthest neighbour an return ot knn
    return farthest neighbour + its knn =k
    https://www.ratml.org/pub/pdf/2017exploiting.pdf Exploiting the structure of furthest neighbor search for
fast approximate results
    unlike tru kfn this function returns furhtest point index and indexes of its k closest neighbours
    :return:
    '''
    if exact == False:
        kf1=approx_kfn(algorithm='ds', calculate_error=False, #using DrusillaSelect approach
                   exact_distances=np.empty([0, 0]), input_model=None, k=1,
                   num_projections=100, num_tables=100, query=query,
                   reference=data)
    else:
        #query = data[:10, :]
        kf1 = kfn(algorithm='dual_tree',  k=1,
        leaf_size=20, percentage=percentage, query=query, random_basis=False,
        reference=data, seed=0, tree_type="kd")
        plt.scatter(kf1['neighbors'], kf1['distances'])
    return kf1



plt.hist(zzz['distances'],50)
#TODO: find mean per cluster and distances between means and compare with this histogram
#TDOD: compute exact for small data and compare
def k_farthest_neib_HD_samples(query, data, sample=10000):
    '''
    similar idea to farthestspoints_sphere
    instead of convex hull find knn and closest to
    the opposite point on the eclosing circle will be the fartherst one
    or just get the hull
    :return:
    '''
    pass

def global_structure_preservation_scores_mean():
    '''
    find average distance (in x-space) of n-farhest neighbours (in y space) per point
    average over all points in data set
    '''
    pass

def global_structure_preservation_scores_emd():
    '''
    same as above nut uses arth mover distance
    '''
    pass
#TODO: demonstrate this by putting 4 balls on one line and by
# rreplacing positions of two central bulbs, thus destroing global
# (topological) structure
#TODO: for mauscript plot PCA elbow plots to show how variance falls
# inside first several PC's in single cell (Shekhar and Levine32)
#modified cosde for LC-WMD
def get_distance_matrix(point_id):
    dm = dataset_dn[point_id] + queries_qn - 2.0 * np.dot(dataset_prep[point_id], qt)
    dm[dm < 0.0] = 0.0
    dm = np.sqrt(dm)
    return dm

def lc_wmd_cost(dist, k): #apparently distance matrix between vocalbulary and query, done by multiplication (modify fun get_distance_matrix as required
    # when s1  = s2 )
    if dist.shape[0] > dist.shape[1]:
        dist = dist.T
    s1 = dist.shape[0]
    s2 = dist.shape[1]
    cost1 = np.mean(dist.min(axis=0))
    if s1 == s2:
        cost2 = np.mean(dist.min(axis=1))
        return max(cost1, cost2)
    k = min(k, int(np.floor(s2/s1)), s2-1)
    remainder = (1./s1) - k*(1./s2)
    pdist = np.partition(dist, k, axis=1)
    cost2 = (np.sum(pdist[:,:k]) * 1./s2) + (np.sum(pdist[:,k]) * remainder)
    return max(cost1, cost2)

def lc_wmd(query, ids, id_result, score_result, to_sort, k_param=1):
    load_query(query)
    k1 = ids.shape[0]
    k2 = result.shape[0]
    for i in range(k1):
        dm = get_distance_matrix(ids[i])# compute actual disctance matrix
        scores[i] = lc_wmd_cost(dm, k_param)
    solver.select_topk(ids, scores, id_result, score_result, to_sort)# most likely do not need this line, since i just need distance not list of closest
    # documents
