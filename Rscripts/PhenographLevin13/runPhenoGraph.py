#To evaluate phenograph on the subset of gates, with cell subsampling.
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
ROOT = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparison'
DATA_DIR = "/benchmark_data_sets/"
DATA = 'Levine_13dim_EXPR_ONLY.fcs'

MANUAL_PHENOGRAPH = "/results/manual/phenoGraph/"
RES_DIR_PHENOGRAPH = "/results/auto/PhenoGraph/"
CALC_NAME = 'Kdependency15Gates30000'

#data=np.loadtxt(ROOT + DATA_DIR + DATA, skiprows = 1, converters = {13: lambda s: s==b'NA' and -1 or s})
meta, data = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=False, reformat_meta=True, output_format='ndarray')

#data[0].keys()
meta = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=True, reformat_meta=True)
meta['_channels_']
SET_NAME = 'Levine_13dim'

#filter the matrix, keeping 15 most prominent populations, see S2 in PG paper
#1	CD11b-_Monocyte_cells
#2	CD11bhi_Monocyte_cells
#7	HSC_cells
#23	Pre-B_I_cells
#24	Pre-B_II_cells
#8	Immature_B_cells
#9	Mature_CD38lo_B_cells ??? = Mature B from figure S2 in PG paper, or add mid as well?
#10	Mature_CD38mid_B_cells  = Mature B from figure S2 in PG paper, or add low as well?
# #ok, add them as one class for now, see how similatr those in http://www.bloodjournal.org/content/128/7/923?sso-checked=true
#class number 25, that way we just have 15 populations as in PG paper
#19	NK_cells
#18	Naive_CD8+_T_cells
#12	Mature_CD8+_T_cells
#17	Naive_CD4+_T_cells
#11	Mature_CD4+_T_cells
#13	Megakaryocyte_cells
#5	Erythroblast_cells
#21	Plasmacytoid_DC_cells

#subset the data
l=np.loadtxt(ROOT + MANUAL_PHENOGRAPH + "true_labels_phenoGraph_" + SET_NAME+ ".txt", skiprows = 1, converters = {0: lambda s: s==b'NA' and -1 or s})
data=np.c_[data, l]
labcol=np.shape(data)[1]-1

IDX= [x for x in range(np.shape(data)[0]) if data[x, labcol] in [1,2,7,23,24,8,9,10,19,18,12,17,11,13,5,21]]
data2= data[IDX,]
np.shape(data2)
#now unify classes 9 and 10 into #25
IDX2 = [x for x in range(np.shape(data2)[0]) if data2[x, labcol] in [9,10]]
data2[IDX2, labcol] = 25
np.shape(data2)
plt.hist(data2[:, labcol], bins=30)
#30 000 random sample:
sampleIDX = np.random.choice(range((np.shape(data2))[0]), size=30000,  replace=False)
data3=data2[sampleIDX,]
data=data3

#run phenograph
res=[]
for i in [15, 30, 45, 60, 75, 90, 105]:
    communities, graph, Q = phenograph.cluster(data[:, :labcol], k=i)
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
           fmt='%f', X = (data[:, ]))

#save 30 000 indices
#if not os.path.exists(ROOT + RES_DIR_PHENOGRAPH + CALC_NAME):
#    os.makedirs(ROOT + RES_DIR_PHENOGRAPH + CALC_NAME)
#np.savetxt(ROOT + MANUAL_PHENOGRAPH + CALC_NAME+"/"+"Python_subset_IDX_" + SET_NAME + ".txt", newline='\n',
#           fmt='%i',  X = sampleIDX)


#NMI score
for k in range(7):
    com=res[k][0]
    print(sklearn.metrics.adjusted_mutual_info_score((data[:, labcol]).astype(int), com))

print(sklearn.metrics.adjusted_mutual_info_score(res[3][0], com))

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

