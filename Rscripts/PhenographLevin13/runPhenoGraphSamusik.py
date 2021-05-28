#To evaluate Phenograph on uncurated data set
import phenograph
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import networkx as nx
import os
import sklearn
import fcsparser



#Set the variables
ROOT = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL'
DATA_DIR = "/benchmark_data_sets/"
DATA = 'Samusik_all_EXPR_ONLY.fcs'

MANUAL_PHENOGRAPH = "/results/manual/phenoGraph/"
RES_DIR_PHENOGRAPH = "/results/auto/phenoGraph/"
CALC_NAME = 'KdependencySamusik_allgatedL1'
SET_NAME = 'Samusik_all'

#data=np.loadtxt(ROOT + DATA_DIR + DATA, skiprows = 1, converters = {13: lambda s: s==b'NA' and -1 or s})
meta, data = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=False, reformat_meta=True, output_format='ndarray')

#data[0].keys()
meta = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=True, reformat_meta=True)
meta['_channels_']

#labels
l=np.loadtxt(ROOT + MANUAL_PHENOGRAPH + "true_labels_phenoGraph_" + SET_NAME + ".txt", skiprows = 1, converters = {0: lambda s: s==b'NA' and -1 or s})
data=np.c_[data, l]
labcol=np.shape(data)[1]-1

IDX= [x for x in range(np.shape(data)[0]) if not np.isnan(data[x, labcol])]
data= data[IDX,]

#run phenograph
res=[]
for i in [15, 30, 45, 60, 75, 90, 105]:
    communities, graph, Q = phenograph.cluster(data[:, :labcol], k=i, n_jobs=5,  primary_metric = 'manhattan')
    res.append([communities, graph, Q])

#plt.hist(communities)

#save true labels
if not os.path.exists(ROOT + MANUAL_PHENOGRAPH + CALC_NAME):
    os.makedirs(ROOT + MANUAL_PHENOGRAPH + CALC_NAME)
np.savetxt(ROOT + MANUAL_PHENOGRAPH + CALC_NAME+"/"+"Python_true_labels_phenoGraph_" + SET_NAME + ".txt", newline='\n',
           fmt='%i', header='label', X = (data[:, labcol]).astype(int))

# save cluster labels
if not os.path.exists(ROOT + RES_DIR_PHENOGRAPH + CALC_NAME):
    os.makedirs(ROOT + RES_DIR_PHENOGRAPH + CALC_NAME)
for k in range(7):
    com=res[k][0]
    np.savetxt(ROOT + RES_DIR_PHENOGRAPH + CALC_NAME+"/"+'k=' + str(k*15+15)
               + "Python_assigned_labels_phenoGraph_" + SET_NAME + ".txt", com, fmt='%d',  header='label', newline='\n' )

#save data
if not os.path.exists(ROOT + RES_DIR_PHENOGRAPH + CALC_NAME):
    os.makedirs(ROOT + RES_DIR_PHENOGRAPH + CALC_NAME)
np.savetxt(ROOT + MANUAL_PHENOGRAPH + CALC_NAME+"/"+"data_" + SET_NAME + ".txt", newline='\n',
           fmt='%f', X = (data[:, ]))


#AMI score
for k in range(7):
    com=res[k][0]
    print(sklearn.metrics.adjusted_mutual_info_score(l, com))

print(sklearn.metrics.adjusted_mutual_info_score(res[3][0], com))
#NMI score
for k in range(7):
    restxt=np.loadtxt(ROOT + RES_DIR_PHENOGRAPH + CALC_NAME + "/" + 'k=' + str(k * 15 + 15)
               + "Python_assigned_labels_phenoGraph_" + SET_NAME + ".txt")
    com=restxt
    print(sklearn.metrics.normalized_mutual_info_score(l, com))



#plt.figure()
#plot_coo_matrix(graph)
#plt.savefig('123.png')


G1=nx.from_numpy_matrix(res[0][1].todense())

norm = mpl.colors.Normalize(vmin=-20, vmax=10)
cmap = plt.cm.hot
m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
node_color = m.to_rgba(communities)

plt.figure()
nx.draw_networkx(G1,  node_size=.1, width=0.01, node_color=Q, with_labels=False)
plt.savefig('123.png')

