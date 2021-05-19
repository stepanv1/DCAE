'''
Generates artificial clusters
and computes objects required to run
DCAE
generate UMAP data and plots for comparison
'''

import numpy as np
import matplotlib.pyplot as plt
import os


from utils_evaluation import  table, find_neighbors, compare_neighbours, \
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, show3d, \
    preprocess_artificial_clusters,  generate_clusters_pentagon

os.chdir('/home/stepan/PycharmProjects/BIOIBFO25L/')

output_dir  = "./Artificial_sets/"

k=30
markers = np.arange(30).astype(str)

# generate clusters with different  branching points (25 topologies)
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])

# sizes of clusters: 124 000 (k*62 000)
for b in list_of_branches:
    aFrame, lbls = generate_clusters_pentagon(num_noisy = 25, branches_loc = b,  sep=3/2, pent_size=2, k=2)
    #preprocess to generate neural netowrk parameters amd neibours for performance metrics estimation,
    #saves all obects in npz
    aFrame, Idx, Dist, Sigma, lbls, neibALL =  preprocess_artificial_clusters(aFrame, lbls, k=30, num_cores=10, outfile=output_dir + 'set_' + str(b) +  '.npz' )
    #save csv
    np.savetxt(output_dir + 'aFrame_' + str(b) + '.csv', aFrame, delimiter=',')
    np.savetxt(output_dir + 'Idx.csv_' + str(b) + '.csv', Idx, delimiter=',')
    np.savetxt(output_dir + 'Dist_' + str(b) + '.csv', Dist, delimiter=',')
    np.savetxt(output_dir + 'Sigma_' + str(b) + '.csv', Sigma, delimiter=',')
    np.savetxt(output_dir + 'lbls.csv_' + str(b) + '.csv', lbls, delimiter=',')
    #data = np.loadtxt('data.csv', delimiter=',')


'''
import seaborn as sns
import umap.umap_ as umap
import plotly.io as pio
pio.renderers.default = "browser"
yl =7
fig, axs = plt.subplots(nrows=8)
sns.violinplot(data=aFrame[lbls==0,:],  ax=axs[0]).set_title('0', rotation=-90, position=(1, 1), ha='left', va='bottom')
axs[0].set_ylim(0, yl)
sns.violinplot(data=aFrame[lbls==1,:],  ax=axs[1]).set_title('1', rotation=-90, position=(1, 2), ha='left', va='center')
axs[1].set_ylim(0, yl)
sns.violinplot(data=aFrame[lbls==2,:],  ax=axs[2]).set_title('2', rotation=-90, position=(1, 2), ha='left', va='center')
axs[2].set_ylim(0, yl)
sns.violinplot(data=aFrame[lbls==3,:],  ax=axs[3]).set_title('3', rotation=-90, position=(1, 2), ha='left', va='center')
axs[3].set_ylim(0, yl)
sns.violinplot(data=aFrame[lbls==4,:],  ax=axs[4]).set_title('4', rotation=-90, position=(1, 2), ha='left', va='center')
axs[4].set_ylim(0, yl)
sns.violinplot(data=aFrame[lbls==5,:],  ax=axs[5]).set_title('5', rotation=-90, position=(1, 2), ha='left', va='center')
axs[5].set_ylim(0, yl)
sns.violinplot(data=aFrame[lbls==6,:],  ax=axs[6]).set_title('6', rotation=-90, position=(1, 2), ha='left', va='center')
axs[6].set_ylim(0, yl)
sns.violinplot(data=aFrame[lbls==-7,:], ax=axs[7]).set_title('7', rotation=-90, position=(1, 2), ha='left', va='center')
axs[7].set_ylim(0, yl)





mapper = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=True).fit(aFrame)
yUMAP =  mapper.transform(aFrame)

from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(aFrame)
yPCA = pca.transform(aFrame)

pca = decomposition.PCA(n_components=10)
pca.fit(aFrame)
yPCA3 = pca.transform(aFrame)



fig = plot2D_cluster_colors(yUMAP, lbls=lbls, msize=5)
fig.show()
fig = plot2D_cluster_colors(yPCA, lbls=lbls, msize=5)
fig.show()
'''








