'''
Generates artificial clusters
and computes objects required to run
DCAE
generate UMAP data and plots for comparison
'''

import numpy as np
import os


from utils_evaluation import  preprocess_artificial_clusters,  generate_clusters_pentagon

from utils_evaluation import preprocess_artificial_clusters_LSH #comment out then running SAUCIE and UMAP scripts

os.chdir('/media/grinek/Seagate/DCAE/')

output_dir  = "/media/grinek/Seagate/Artificial_sets/"

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
# experiments with 11 spheres data from Moors et al 'Topologiacl autoencoder'
'''
Trying out Tadasets library for generating topological synthetic datasets: 
sphere 
'''
import numpy as np
# import tadasets
import matplotlib
import matplotlib.pyplot as plt
from tadasets.dimension import embed
import plotly.io as pio
pio.renderers.default = "browser"


def dsphere(n=100, d=2, r=1, noise=None, ambient=None):
    """
    Sample `n` data points on a d-sphere.
    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """
    data = np.random.randn(n, d+1)

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)



    return data

def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    # it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance = 10 / np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d + 1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres - 1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 10 * n_samples  # int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r * 5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        plt.show()

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    return dataset, labels

dataset, labels  = create_sphere_dataset(n_samples=10000, d=100, r=5, plot=False, seed=42)

from sklearn import decomposition
from utils_evaluation import plot2D_cluster_colors
pca = decomposition.PCA(n_components=2)
pca.fit(dataset)
yPCA = pca.transform(dataset)
fig = plot2D_cluster_colors(yPCA, lbls=labels, msize=5)
fig.show()

b = 'spheresbig'
aFrame, Idx, Dist, Sigma, lbls, neibALL =  preprocess_artificial_clusters_LSH(dataset, labels, k=30, num_cores=16, outfile='/media/grinek/Seagate/Artificial_sets/Art_set25/' + 'set_' + str(b) +  '.npz' )

pca.fit(dataset)
yPCA = pca.transform(aFrame)
fig = plot2D_cluster_colors(yPCA, lbls=labels, msize=1)
fig.show()

#save csv
output_dir ='/media/grinek/Seagate/Artificial_sets/Art_set25/'
np.savetxt(output_dir + 'aFrame_' + str(b) + '.csv', aFrame, delimiter=',')
np.savetxt(output_dir + 'Idx.csv_' + str(b) + '.csv', Idx, delimiter=',')
np.savetxt(output_dir + 'Dist_' + str(b) + '.csv', Dist, delimiter=',')
np.savetxt(output_dir + 'Sigma_' + str(b) + '.csv', Sigma, delimiter=',')
np.savetxt(output_dir + 'lbls.csv_' + str(b) + '.csv', lbls, delimiter=',')






