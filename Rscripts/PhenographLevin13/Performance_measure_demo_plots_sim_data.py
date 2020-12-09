import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d
#get a subsample of Levine data and create artificial data with it


PAPERPLOTS  = './PAPERPLOTS/'
#data, color = datasets.make_blobs(n_samples=10000, centers=20, n_features=30,
#                   random_state=0)

data, color = datasets.make_classification(n_samples=20000, n_features=15,  n_informative=5, n_redundant=0, n_repeated=0,
                n_classes=3, n_clusters_per_class=1, weights=None, flip_y=0.5, class_sep=2.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=12345)


import umap.umap_ as umap
mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(data)
y0 =  mapper.transform(data)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
y0= scaler.fit_transform(y0)
import copy
y=copy.deepcopy(y0)

cdict = {0: 'red', 1: 'blue', 2: 'green'}
#plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
dataset = pd.DataFrame()
dataset['x'] = y0[:, 0]
dataset['y'] = y0[:, 1]
dataset['color'] = [str(x) for x in color]
plt.scatter(y0[:, 0], y0[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
#sns.scatterplot(data=dataset, x="x", y="y", hue="color")
#lets overlap and split our data
y0[(y0[:,0]>=.6) & (y0[:,1]>=0.5), 1]=y0[(y0[:,0]>=.6) & (y0[:,1]>=.5), 1]+0.5
#plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
y0[(y0[:,0]<0.6) & (y0[:,1]>0.5), 1]=y0[(y0[:,0]<0.6) & (y0[:,1]>0.5), 1]-0.73
plt.scatter(y0[:, 0], y0[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)


data =  scaler.fit_transform(data)

neib_data = find_neighbors(data, 30, metric='euclidean')['idx']
onetomany_score = neighbour_onetomany_score(y0, neib_data, kmax=30, num_cores=12)[1]
onetomany_score[29,:]
marker_similarity_score = neighbour_marker_similarity_score_per_cell(y0, data, kmax=30, num_cores=12)[1]

discontinuity, manytoone = get_wsd_scores(data, y0, 30, num_meandist=10000, compute_knn_x=False, x_knn=neib_data)



vmax1 = np.percentile(discontinuity,95)
vmax2 = np.percentile(manytoone,95)
vmax3=np.percentile(onetomany_score[29,:],95)
vmax4 = np.percentile(marker_similarity_score[29,:],95)
vmin1 = np.percentile(discontinuity,5)
vmin2 = np.percentile(manytoone,5)
vmin3=np.percentile(onetomany_score[29,:],5)
vmin4 = np.percentile(marker_similarity_score[29,:],5)

PAPERPLOTS  = './PAPERPLOTS/'
sz=0.1
sns.set_style("white")
fig = plt.figure(figsize=(10, 10))
sbpl1= plt.subplot(3, 2, 1)
plt.title("UMAP")
plt.scatter(y[:, 0], y[:, 1], c=color, s=sz, cmap=plt.cm.Spectral)

sbpl2= plt.subplot(3, 2, 2)
plt.title("Broken UMAP")
plt.scatter(y0[:, 0], y0[:, 1], c=color, s=sz, cmap=plt.cm.Spectral)
#plt.axis("off")

sbpl3 = plt.subplot(3, 2, 3)
plt.title("discontinuity")
img= plt.scatter(y0[:, 0], y0[:, 1], c=discontinuity,
            vmax=vmax1, vmin=vmin1,
            s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax1)

sbpl4 = plt.subplot(3, 2, 4)
plt.title("manytoone")
img = plt.scatter(y0[:, 0], y0[:, 1], c=manytoone,
                  vmax=vmax2, vmin=vmin2,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax2)

plt.subplot(3, 2, 5)
sbpl5 = plt.title("onetomany")
img = plt.scatter(y0[:, 0], y0[:, 1], c=onetomany_score[29,:]
, vmax=vmax3, vmin=vmin3,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax3)

sbpl6 = plt.subplot(3, 2, 6)
plt.title("marker similarity")
img = plt.scatter(y0[:, 0], y0[:, 1], c=marker_similarity_score[29,:]
, vmax=vmax4,  vmin=vmin4,  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax4)

#cbax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
#cb = plt.colorbar(img, cax=cbax)
#cb.outline.set_linewidth(0.)
#plt.clim(0., vmax)

plt.savefig(PAPERPLOTS  + 'Illustration_' + 'performance_measures.png')
plt.show()

plt.clf()

plt.hist(onetomany_score[29,:],100)
plt.hist(discontinuity,100)


plt.hist(marker_similarity_score[29,:],300)
plt.hist(manytoone,100)