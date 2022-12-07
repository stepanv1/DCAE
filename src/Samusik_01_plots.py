'''
Plots for Samusik_01 data
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import rcParams
import numpy as np
import os
import seaborn as sns
from utils_evaluation import table
from matplotlib.colors import rgb2hex

k = 30
epochs_list = [1000]
coeffCAE = 1
coeffMSE = 1
batch_size = 128
lam = 1
alp = 0.5
m = 10
patience = 1000
min_delta = 1e-4

ID = 'DCAE' + '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)

os.chdir('/media/grinek/Seagate/DCAE/')
DATA_ROOT = '/media/grinek/Seagate/'
DATA_DIR = DATA_ROOT + 'CyTOFdataPreprocess/'
source_dir = DATA_ROOT + 'Real_sets/'
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
                  'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz', 'Shenkareuclid_shifted.npz', 'Samusik_01.npz']
PLOTS = DATA_ROOT + "Real_sets/PLOTS/"
z_dir = DATA_ROOT + 'Real_sets/DCAE_output/'
output_dir = DATA_ROOT + "Real_sets/DCAE_output/Performance/"
temp_dir  =  DATA_ROOT + "Real_sets/DCAE_output/Performance/temp/"

bl_index  = [0,1,2,3]
#azymuth, elevaation , position
camera_positions = [[[38,10,0], [174,79,0], [-122,9,0]], [[101,-42,0], [3,-7,0], [-51,30,0]], [[-145,-57,0], [-160,15,0], [7,5,0]],
                    [[-100,-39,0], [0,0,0], [0,0,0]]]
epochs = 1000
#idx = bl_index[3]
unassigned_lbls = ['"unassigned"', '"Unassgined"', '-1', '-1']


print(output_dir)
bl = list_of_inputs[3]
print(bl)
# read data
infile = DATA_DIR + bl
npzfile = np.load(infile, allow_pickle=True)
lbls = npzfile['lbls']
table(lbls)

# read DCAE output
npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz', allow_pickle=True)
z = npz_res['z']

# get progenitors
df = z[np.isin(lbls, ['HSC', 'MPP', 'CLP',  'CMP',  'MEP', 'GMP']), :]
lb=lbls[np.isin(lbls,  ['HSC', 'MPP', 'CLP',  'CMP',  'MEP', 'GMP'])]

cl = np.unique(lb)

#use this to find a good angle
from mpl_toolkits.mplot3d import Axes3D

dpi = 350
rcParams['savefig.dpi'] = dpi
sz=10
fig = plt.figure(dpi=dpi, figsize=(10, 10))
ax = Axes3D(fig)
loc = plticker.MultipleLocator(base=0.5)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.zaxis.set_major_locator(loc)
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
sns.reset_orig()

nM = len(np.unique(lb))
palette = sns.color_palette("husl", nM)
colors = np.array([rgb2hex(palette[i]) for i in range(len(palette))])

#['CLP', 'CMP', 'GMP', 'HSC', 'MEP', 'MPP']
colors =['violet','greenyellow', 'seagreen', 'black', 'lime', 'cornflowerblue']
sz=5
groups = []
for i in range(len(cl)):
    groups.append(ax.scatter(xs=df[:,0][lb==cl[i]], ys=df[:,1][lb==cl[i]], zs=df[:,2][lb==cl[i]], c = colors[i],  s=sz, alpha=0.3))

#check cluster structure near root of
# from sklearn.decomposition import PCA
# import plotly.io as pio
# pio.renderers.default = "browser"
# import plotly.express as px
#
# infile = '/media/grinek/Seagate/CyTOFdataPreprocess/' + bl
# npzfile = np.load(infile, allow_pickle=True)
# aFrame =npzfile['aFrame']
# markers = npzfile['markers']
# npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz', allow_pickle=True)
# df =aFrame[np.isin(lbls,  ['CLP', 'MPP', 'CMP']), :]
# pca = PCA(n_components =4)
# components = pca.fit_transform(df)
# fig = px.scatter_matrix(
#     components,
#     color=lbls[np.isin(lbls,  ['CLP', 'MPP','CMP'])]
# )
# fig.update_traces(diagonal_visible=False, marker={'size': 5})
# fig.show()
#
# df = aFrame[np.isin(lbls, ['CLP', 'MPP', 'CMP', 'HSC', 'MEP', 'GMP']), :]
# pca = PCA(n_components=4)
# components = pca.fit_transform(df)
# fig = px.scatter_matrix(
#     components,
#     color=lbls[np.isin(lbls,  ['CLP', 'MPP', 'CMP', 'HSC', 'MEP', 'GMP'])]
# )
# fig.update_traces(diagonal_visible=False, marker={'size': 5})
# fig.show()
#
# df = aFrame[np.isin(lbls, [ 'MPP', 'CMP',  'GMP',  'IgD- IgMpos B cells', 'IgDpos IgMpos B cells',
#                             'IgM- IgD- B-cells', 'B-cell Frac A-C (pro-B cells)' ]), :]
# pca = PCA(n_components=4)
# components = pca.fit_transform(df)
# fig = px.scatter_matrix(
#     components,
#     color=lbls[np.isin(lbls, [ 'MPP', 'CMP',  'GMP',  'IgD- IgMpos B cells', 'IgDpos IgMpos B cells',
#                                'IgM- IgD- B-cells' ,'B-cell Frac A-C (pro-B cells)'])]
# )
# fig.update_traces(diagonal_visible=False, marker={'size': 5})
# fig.show()

ax.view_init(azim=camera_positions[3][0][0],  elev=camera_positions[3][0][1])
ax.set_rasterized(True)

order = [3,5,0,1,2,4]
lgnd = ax.legend([groups[idx] for idx in order],[cl[idx] for idx in order], loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=24, markerscale=6)
#lgnd = ax.legend(groups, cl, loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=16, markerscale=30)
for handle in lgnd.legendHandles:
    handle.set_sizes([30.0])
plt.savefig( PLOTS + list_of_inputs[3] +  '_progenitors_paper_DCAE.eps', dpi= dpi, format='eps', bbox_inches='tight')
plt.savefig( PLOTS + list_of_inputs[3] +  '_progenitors_paper_DCAE.tif', dpi= dpi, format='tif', bbox_inches='tight')



##################################################################################################
#2D plots of UMAP and SAUCIE
#3 data sets in one row
z_UMAP  = DATA_ROOT + "Real_sets/UMAP_output/"
z_SAUCIE = DATA_ROOT + "Real_sets/SAUCIE_output/"
idx=3
print(output_dir)
bl = list_of_inputs [idx]
print(bl)
infile = DATA_DIR + bl
npzfile = np.load(infile, allow_pickle=True)
lbls = npzfile['lbls']

dpi = 350
rcParams['savefig.dpi'] = dpi
sz = 1

fig = plt.figure(dpi=dpi, figsize=(10, 5))
# plot UMAP
npz_res = np.load(z_UMAP + str(bl) + '_UMAP_rep_2D.npz', allow_pickle=True)
z = npz_res['z']

# get progenitors
df = z[np.isin(lbls, ['HSC', 'MPP', 'CLP',  'CMP',  'MEP', 'GMP']), :]
lb=lbls[np.isin(lbls,  ['HSC', 'MPP', 'CLP',  'CMP',  'MEP', 'GMP'])]

cl = np.unique(lb)

ax = fig.add_subplot(1, 2, 1)
loc = plticker.MultipleLocator(base=5)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

groups = []
for i in range(len(cl)):
    groups.append(
            ax.scatter(x=df[:, 0][lb == cl[i]], y=df[:, 1][lb == cl[i]], c=colors[i], s=sz))
plt.grid(True, linewidth=0.5)

# plot SAUCIE
infile = DATA_DIR + bl
npzfile = np.load(infile, allow_pickle=True)
lbls = npzfile['lbls']
npz_res = np.load(z_SAUCIE + '/' + str(bl) + '_SAUCIE_rep_2D.npz', allow_pickle=True)
z = npz_res['z']

# get progenitors
df = z[np.isin(lbls, ['HSC', 'MPP', 'CLP',  'CMP',  'MEP', 'GMP']), :]
lb=lbls[np.isin(lbls,  ['HSC', 'MPP', 'CLP',  'CMP',  'MEP', 'GMP'])]

cl = np.unique(lb)
ax = fig.add_subplot(1, 2, 2)
loc = plticker.MultipleLocator(base=1)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

groups = []
for i in range(len(cl)):
    if cl[i] != unassigned_lbls[idx]:
        groups.append(
            ax.scatter(x=df[:, 0][lb == cl[i]], y=df[:, 1][lb == cl[i]], c=colors[i], s=sz))
plt.grid(True, linewidth=0.5)
plt.savefig(PLOTS + list_of_inputs[idx] + 'UMAP_SAUCIE_progentitors.eps', dpi=dpi, format='eps')
plt.savefig(PLOTS + list_of_inputs[idx] + 'UMAP_SAUCIE_progentitors.tif', dpi=dpi, format='tif')


