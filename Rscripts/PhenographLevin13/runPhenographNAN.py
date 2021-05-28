#Phenograph runs on full sets with NAN included
#To evaluate phenograph on the full set of gates, without cell subsampling as well
import phenograph
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import networkx as nx
import os
import sklearn
import fcsparser
import pandas


#Data-Driven Phenotypic Dissection of AML Reveals
#Progenitor-like Cells that Correlate with Prognosis
#, Levine
#test the performance of PhenoGraph in the settings
#of the PhenoGraph paper, "Benchmark Data Set 1 ".
#See the pictures 2A, S2B, DAta S1B-C frpm

#Only cells that were assigned to prominent cell types by manual
#gating were retained. 30,000 cells were sampled randomly from these gates.
#List of cell types left in the data-sets:  ______________????
#Total 15 populations, fig. S2.

#Set the variables
ROOT = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL'
DATA_DIR = "/benchmark_data_sets/"
DATA = 'Samusik_all.fcs'

MANUAL_PHENOGRAPH = "/results/manual/phenoGraph/"
RES_DIR_PHENOGRAPH = "/results/auto/PhenoGraph/"
CALC_NAME = 'ManhattanSamusik_NAN'
SET_NAME = 'Samusik_all'
#data=np.loadtxt(ROOT + DATA_DIR + DATA, skiprows = 1, converters = {13: lambda s: s==b'NA' and -1 or s})
meta, data = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=False, reformat_meta=True, output_format='ndarray')

#data[0].keys()
meta = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=True, reformat_meta=True)
meta['_channels_']

#subset the data
#l=np.loadtxt(ROOT + MANUAL_PHENOGRAPH + "true_labels_phenoGraph_" + SET_NAME + ".txt", skiprows = 1, converters = {0: lambda s: s==b'NA' and -1 or s})
#data=np.c_[data, l]
labcol=np.shape(data)[1]-1

res=[]
for i in [15, 30, 45]:
    communities, graph, Q = phenograph.cluster(data[:, :labcol], k=i, n_jobs=3, nn_method='brute', primary_metric='manhattan')
    res.append([communities, graph, Q])

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
           fmt='%f', X = (data))


#np.loadtxt(ROOT + DATA_DIR + DATA, skiprows = 1, converters = {13: lambda s: s==b'NA' and -1 or s})
#NMI score
mask = data[:, labcol]!=0
for k in range(7):
    com=np.loadtxt(ROOT + RES_DIR_PHENOGRAPH + CALC_NAME+"/"+'k=' + str(k*15+15)
               + "Python_assigned_labels_phenoGraph_" + SET_NAME + ".txt", skiprows = 1)
    print(sklearn.metrics.normalized_mutual_info_score((data[:, labcol][mask]).astype(int), com[mask]))

print(sklearn.metrics.adjusted_mutual_info_score((data[:, labcol][mask]).astype(int), com[mask]))

#plt.figure()
#plot_coo_matrix(graph)
#plt.savefig('123.png')


G1=nx.from_numpy_matrix(graph.todense())

norm = mpl.colors.Normalize(vmin=-20, vmax=10)
cmap = plt.cm.hot
m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
node_color = m.to_rgba(communities)

plt.figure()
nx.draw_networkx(G1,  node_size=.1, width=0.01, node_color=Q, with_labels=False)
plt.savefig('123.png')

