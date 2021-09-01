import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from utils_evaluation import compute_f1, table, find_neighbors,  neighbour_onetomany_score_normalized, \
    get_wsd_scores_normalized, neighbour_marker_similarity_score_per_cell
#get a subsample of Levine data and create artificial data with it
#import ot_estimators
import os

os.chdir('/home/grinek/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/grinek/Seagate/'
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


#data =  scaler.fit_transform(data)

neib_data = find_neighbors(data, 30, metric='euclidean')['idx']
onetomany_score = neighbour_onetomany_score_normalized(y0, neib_data, kmax=30, num_cores=12)[1]
onetomany_score[29,:]
marker_similarity_score = neighbour_marker_similarity_score_per_cell(y0, data, kmax=30, num_cores=12)[1]

discontinuity, manytoone = get_wsd_scores_normalized(data, y0, 30, num_meandist=10000, compute_knn_x=False, x_knn=neib_data)
plt.hist(manytoone,25)
plt.hist(marker_similarity_score[29,:],100)

vmax1 = np.percentile(discontinuity,95)
vmax2 = np.percentile(manytoone,95)
vmax3=np.percentile(onetomany_score[29,:],95)
vmax4 = np.percentile(marker_similarity_score[29,:],95)
vmin1 = np.percentile(discontinuity,5)
vmin2 = np.percentile(manytoone,5)
vmin3=np.percentile(onetomany_score[29,:],5)
vmin4 = np.percentile(marker_similarity_score[29,:],5)

import matplotlib
matplotlib.use('PS')

PAPERPLOTS  = DATA_ROOT + 'PLOTS/'
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
#plt.clim(vmin1, vmax1)

sbpl4 = plt.subplot(3, 2, 4)
plt.title("manytoone")
img = plt.scatter(y0[:, 0], y0[:, 1], c=manytoone,
                  vmax=vmax2, vmin=vmin2,
#vmin=0.15, vmax=0.8,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
#plt.clim(vmin2, vmax2)

plt.subplot(3, 2, 5)
sbpl5 = plt.title("LSSS")
img = plt.scatter(y0[:, 0], y0[:, 1], c=onetomany_score[29,:]
, vmax=vmax3, vmin=vmin3,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
#plt.clim(vmin3, vmax3)

sbpl6 = plt.subplot(3, 2, 6)
plt.title("MSS")
img = plt.scatter(y0[:, 0], y0[:, 1], c=marker_similarity_score[29,:]
, vmax=vmax4,  vmin=vmin4,  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
#plt.clim(vmin1, vmax4)

#cbax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
#cb = plt.colorbar(img, cax=cbax)
#cb.outline.set_linewidth(0.)
#plt.clim(0., vmax)

plt.savefig(PAPERPLOTS  + 'Illustration_' + 'performance_measures_normalized.eps', format='eps', dpi = 350)
plt.close()


plt.hist(onetomany_score[29,:],100)
plt.hist(discontinuity,100)


plt.hist(marker_similarity_score[29,:],300)
plt.hist(manytoone,100)

#############################################################
# try reshape
mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(data)
y2 =  mapper.transform(data)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
y2= scaler.fit_transform(y2)
y=y2.copy()

cdict = {0: 'red', 1: 'blue', 2: 'green'}
#plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
dataset = pd.DataFrame()
dataset['x'] = y0[:, 0]
dataset['y'] = y0[:, 1]
dataset['color'] = [str(x) for x in color]
#plt.scatter(y0[:, 0], y0[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
#sns.scatterplot(data=dataset, x="x", y="y", hue="color")
#lets overlap and split our data
#2[(y2[:,0]>=.6) & (y2[:,1]>=0.5), 0]=y2[(y2[:,0]>=.6) & (y2[:,1]>=.5), 0]*2
#plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
y2[(y2[:,0]<0.6) & (y2[:,1]>0.5), 1]=((y2[(y2[:,0]<0.6) & (y2[:,1]>0.5), 1]-0.5)*10)**2 + 0.5
y2[(y2[:,0]<0.6) & (y2[:,1]<0.5), 1]=-((y2[(y2[:,0]<0.6) & (y2[:,1]<0.5), 1]-0.5)*10)**2 + 0.5
y2[(y2[:,0]<0.08) & (y2[:,0]<0.6) , 0] = ((y2[(y2[:,0]<0.08) & (y2[:,0]<0.6) , 0]-0.08)*1)**2 + 0.08
y2[(y2[:,0]>0.08) & (y2[:,0]<0.6), 0]=-((y2[(y2[:,0]>0.08) & (y2[:,0]<0.6), 0]-0.08)*1)**2 + 0.08

plt.scatter(y2[:, 0], y2[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)



data =  scaler.fit_transform(data)

neib_data = find_neighbors(data, 30, metric='euclidean')['idx']
onetomany_score = neighbour_onetomany_score_normalized(y0, neib_data, kmax=30, num_cores=12)[1]
onetomany_score[29,:]
marker_similarity_score = neighbour_marker_similarity_score_per_cell(y0, data, kmax=30, num_cores=12)[1]

discontinuity, manytoone = get_wsd_scores_normalized(data, y0, 30, num_meandist=10000, compute_knn_x=False, x_knn=neib_data)



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
plt.scatter(y2[:, 0], y2[:, 1], c=color, s=sz, cmap=plt.cm.Spectral)
#plt.axis("off")

sbpl3 = plt.subplot(3, 2, 3)
plt.title("discontinuity")
img= plt.scatter(y2[:, 0], y2[:, 1], c=discontinuity,
            vmax=vmax1, vmin=vmin1,
            s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax1)

sbpl4 = plt.subplot(3, 2, 4)
plt.title("manytoone")
img = plt.scatter(y2[:, 0], y2[:, 1], c=manytoone,
                  vmax=vmax2, vmin=vmin2,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax2)

plt.subplot(3, 2, 5)
sbpl5 = plt.title("onetomany")
img = plt.scatter(y2[:, 0], y2[:, 1], c=onetomany_score[29,:]
, vmax=vmax3, vmin=vmin3,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax3)

sbpl6 = plt.subplot(3, 2, 6)
plt.title("marker similarity")
img = plt.scatter(y2[:, 0], y2[:, 1], c=marker_similarity_score[29,:]
, vmax=vmax4,  vmin=vmin4,  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax4)

#cbax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
#cb = plt.colorbar(img, cax=cbax)
#cb.outline.set_linewidth(0.)
#plt.clim(0., vmax)

#plt.savefig(PAPERPLOTS  + 'Illustration_' + 'performance_measures.png')
plt.show()

plt.clf()

data =  scaler.fit_transform(data)
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
y0[(y0[:,0]>=.6) & (y0[:,1]>=0.5), 1]=y0[(y0[:,0]>=.6) & (y0[:,1]>=.5), 1]-0.9
#plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
y0[(y0[:,0]<0.6) & (y0[:,1]>0.5), 1]=y0[(y0[:,0]<0.6) & (y0[:,1]>0.5), 1]-2
plt.scatter(y0[:, 0], y0[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)

neib_data = find_neighbors(data, 30, metric='euclidean')['idx']

d_dis, d_ms = delta_wsd_scores(x=data, y= y, idx=neib_data,  kmax=30, num_cores=12)

vmax1 = np.percentile(discontinuity,95)
vmax2 = np.percentile(manytoone,95)
vmax3=np.percentile(d_dis[1][29,:],95)
vmax4 = np.percentile(d_ms[1],95)
vmin1 = np.percentile(discontinuity,5)
vmin2 = np.percentile(manytoone,5)
vmin3=np.percentile(d_dis[1][29,:],5)
vmin4 = np.percentile(d_ms[1][29,:],5)

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
sbpl5 = plt.title("delta_discontinuty")
img = plt.scatter(y0[:, 0], y0[:, 1], c=d_dis[1][29,:]
, vmax=vmax3, vmin=vmin3,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax3)

sbpl6 = plt.subplot(3, 2, 6)
plt.title("marker similarity")
img = plt.scatter(y0[:, 0], y0[:, 1], c=d_ms[1][29,:]
, vmax=vmax4,  vmin=vmin4,  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax4)

#cbax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
#cb = plt.colorbar(img, cax=cbax)
#cb.outline.set_linewidth(0.)
#plt.clim(0., vmax)

#plt.savefig(PAPERPLOTS  + 'Illustration_delta_measures' + 'performance_measures.png')
plt.show()
plt.hist(onetomany_score[29,:],50)
plt.hist(discontinuity,50)
plt.hist(d_dis[1][29,:],50)
5


# playing with data

data, color = datasets.make_classification(n_samples=100, n_features=5,  n_informative=5, n_redundant=0, n_repeated=0,
                n_classes=3, n_clusters_per_class=1, weights=None, flip_y=0.5, class_sep=2.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=12345)


data2, color2 = datasets.make_classification(n_samples=100, n_features=5,  n_informative=5, n_redundant=0, n_repeated=0,
                n_classes=3, n_clusters_per_class=1, weights=None, flip_y=0.5, class_sep=2.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=1)


import ot
distmat = ot.dist(data, data2)
a = ot.unif(len(data))
b = ot.unif(len(data2))
#this will do
[ot.emd2(a, b, distmat/np.max(distmat), numItermax = 1000, processes=1 ) for x in range(10000)]# 0.06898100585538322

ot.sinkhorn2(a,b,distmat/np.max(distmat), 0.001, numItermax = 50, processes=1) # entropic regularized OT 0.06783081
ot.bregman.empirical_sinkhorn2(data/np.max(distmat), data2/np.max(distmat), 0.001, verbose=False, numIterMax=100) #29.15664872
zzz=ot.bregman.greenkhorn(a, b, distmat/np.max(distmat), 1, numItermax=10000)
np.sum(zzz*distmat)

from pyemd import emd

first_histogram = np.array([0.0, 1.0])
second_histogram = np.array([5.0, 3.0])
emd(first_histogram, second_histogram, distance_matrix)

emd(a, b, distmat)
