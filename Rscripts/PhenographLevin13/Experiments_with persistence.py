# data wrangling
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List
from PIL import Image
from hepml.core import download_dataset
from scipy import ndimage
from sklearn import datasets

# tda magic
from gtda.homology import VietorisRipsPersistence, CubicalPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.plotting import plot_heatmap, plot_point_cloud, plot_diagram
from gtda.pipeline import Pipeline
from hepml.core import make_point_clouds, load_shapes
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d

# dataviz
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


# 3D shapes
download_dataset("shapes.zip")
# diagrams for circles, spheres, tori
download_dataset("diagrams_basic.pkl")
# diagrams for real-world objects
download_dataset("diagrams.pkl")
# computer vision
download_dataset("Cells.jpg")
download_dataset("BlackHole.jpg")
# uncomment to unzip the shapes folder
DATA = Path('../data/')

# To get started, let's generate a synthetic dataset of noisy circles, spheres, and tori,
# where the effect of noise is to displace the points that sample the surfaces by a random amount in a random direction:
point_clouds_basic, labels_basic = make_point_clouds(n_samples_per_shape=10, n_points=20, noise=0.5)
point_clouds_basic.shape, labels_basic.shape

# expect circle
plot_point_cloud(point_clouds_basic[0]).show()
# expect sphere
plot_point_cloud(point_clouds_basic[10]).show()
# expect torus
plot_point_cloud(point_clouds_basic[-1]).show()

# track connected components, loops, and voids
homology_dimensions = [0, 1, 2]
# calculating H2 persistence is memory intensive - see below to use precomputed arrays
persistence = VietorisRipsPersistence(metric="euclidean", homology_dimensions=homology_dimensions, n_jobs=6)
diagrams_basic = persistence.fit_transform(point_clouds_basic)
DATA = Path("../data")
#with open(DATA / "diagrams_basic.pkl", "wb") as f:
#    pickle.dump(diagrams_basic, f)
with open(DATA / "diagrams_basic.pkl", "rb") as f:
    diagrams_basic = pickle.load(f)

diagrams_basic.shape
# circle
import plotly.io as pio
pio.renderers.default = "browser"
fig  = plot_diagram(diagrams_basic[0])
fig.show()
# sphere
plot_diagram(diagrams_basic[10]).show()
# torus
plot_diagram(diagrams_basic[-1]).show()

# From persistent diagrams to machine learning features
# Although persistence diagrams are useful descriptors of the data, they cannot be used directly for machine learning applications. This
# is because different persistence diagrams may have different numbers of points, and basic operations like the addition and multiplication of diagrams are not well-defined.
persistent_entropy = PersistenceEntropy()
# calculate topological feature matrix
X_basic = persistent_entropy.fit_transform(diagrams_basic)
# expect shape - (n_point_clouds, n_homology_dims)
X_basic.shape
plot_point_cloud(X_basic).show()

data, color = datasets.make_classification(n_samples=2000, n_features=3,  n_informative=3, n_redundant=0, n_repeated=0,
                n_classes=3, n_clusters_per_class=1, weights=None, flip_y=0.5, class_sep=5.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=12345)
table(color)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(data)

#homology_dimensions = [0, 1, 2]
# calculating H2 persistence is memory intensive - see below to use precomputed arrays
import gudhi as gd
from scipy.spatial.distance import pdist, squareform
D0 = squareform(pdist(data))
np.max(D0)
np.mean(D0)
np.min(D0)
skeleton_protein0 = gd.RipsComplex(
    points = data,
    max_edge_length = 0.1
)

Rips_simplex_tree_protein0 = skeleton_protein0.create_simplex_tree(max_dimension = 2)
BarCodes_Rips0 = Rips_simplex_tree_protein0.persistence()
#np.array(BarCodes_Rips0)
#plt.hist(BarCodes_Rips0[:, 1])

for i in range(20):
    print(BarCodes_Rips0[i])
persistence_intervals_in_0 = Rips_simplex_tree_protein0.persistence_intervals_in_dimension(0)
persistence_intervals_in_0 = np.array(persistence_intervals_in_0 )
#plt.hist(persistence_intervals_in_0[:1999, 1],  50)
gd.plot_persistence_diagram(BarCodes_Rips0)
gd.plot_persistence_barcode(BarCodes_Rips0)

# create umap representation and fund its parsistence
import umap.umap_ as umap
mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(data)
y0 =  mapper.transform(data)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(y0)
plt.scatter(y0[:, 0], y0[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
D0 = squareform(pdist(y0))
np.max(D0)
np.mean(D0)
skeleton_y0 = gd.RipsComplex(
    points = y0,
    max_edge_length = 0.05
)

Rips_simplex_tree_y0 = skeleton_y0.create_simplex_tree(max_dimension = 2)
BarCodes_Rips_y0 = Rips_simplex_tree_y0.persistence()
#np.array(BarCodes_Rips0)
#plt.hist(BarCodes_Rips0[:, 1])

for i in range(20):
    print(BarCodes_Rips_y0[i])
persistence_intervals_in_y0 = Rips_simplex_tree_y0.persistence_intervals_in_dimension(0)
persistence_intervals_in_y0 = np.array(persistence_intervals_in_y0 )
#plt.hist(persistence_intervals_in_y0[:1999, 1],  50)
gd.plot_persistence_diagram(BarCodes_Rips_y0)
gd.plot_persistence_barcode(BarCodes_Rips_y0)

# find distance
gd.bottleneck_distance(persistence_intervals_in_y0, persistence_intervals_in_0)
gd.bottleneck_distance(persistence_intervals_in_y0, persistence_intervals_in_0, 0.001)



#ips_simplex_tree_protein0 = skeleton_protein0.create_simplex_tree(max_dimension = 2)
#Now we can compute persistence on the simplex tree structure using the persistence() method of the simplex tree class:

In [5]:
BarCodes_Rips0 = Rips_simplex_tree_protein0.persistence()




persistence = VietorisRipsPersistence(metric="euclidean", homology_dimensions=homology_dimensions, n_jobs=6)
diagrams_basic = persistence.fit_transform(data)
DATA = Path("../data")
#with open(DATA / "diagrams_basic.pkl", "wb") as f:
#    pickle.dump(diagrams_basic, f)
with open(DATA / "diagrams_basic.pkl", "rb") as f:
    diagrams_basic = pickle.load(f)

diagrams_basic.shape





