'''
create 2D projections of DCAE, UMAP and SAUCIE outputs for the manuscript
artificial sets
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import os
import seaborn as sns

os.chdir('/home/grinek/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
PLOTS = DATA_ROOT + "Artificial_sets/PLOTS/"
z_dir = DATA_ROOT + 'Artificial_sets/DCAE_output/'

list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
#parameters of run
k = 30
epochs_list = [500]
coeffCAE = 1
coeffMSE = 1
batch_size = 128
lam = 0.1
alp = 0.2
m = 10
patience = 500
min_delta = 1e-4
g=0.1
epochs=500
ID = 'clip_grad_exp_MDS' + '_g_'  + str(g) +  '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)

bl_index  = [0,1,2]
#azymuth, elevaation , position
camera_positions = [[-88.0,-1.0,0], [-163.0,155.0,0]]

bl = '(3, 1)'
print(bl)
# read data
infile = source_dir + 'set_' + str(bl) + '.npz'
npzfile = np.load(infile)
lbls = npzfile['lbls']
lbls =np.abs(lbls)
# read DCAE output
npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz')
z = npz_res['z']
lb = lbls
cl = np.unique(lb)
smpl = np.random.choice(range(z.shape[0]), size=10000, replace=False)
lb = lb[smpl]
z = z[smpl,:]

#z = z[:10000,:]
#lb= lb[:10000]
from matplotlib import rcParams
dpi = 350
rcParams['savefig.dpi'] = dpi
sz=0.0001
fig = plt.figure(dpi = dpi, figsize=(10,6))
# First subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
loc = plticker.MultipleLocator(base=0.5)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.zaxis.set_major_locator(loc)
fig.suptitle('DCAE')
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
sns.reset_orig()
colors = sns.color_palette("husl", n_colors=len(cl))
groups = []
for i in range(len(cl)):
    groups.append(ax.scatter(xs=z[:,0][lb==cl[i]], ys=z[:,1][lb==cl[i]], zs=z[:,2][lb==cl[i]], c = colors[i],  s=sz))
    #ax.legend()
ax.view_init(azim=camera_positions[0][0],  elev=camera_positions[0][1])
# Second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.zaxis.set_major_locator(loc)
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

sns.reset_orig()
colors = sns.color_palette("husl", n_colors=len(cl))
groups = []
for i in range(len(cl)):
    groups.append(
        ax.scatter(xs=z[:, 0][lb == cl[i]], ys=z[:, 1][lb == cl[i]], zs=z[:, 2][lb == cl[i]], c=colors[i], s=sz))
    # ax.legend()
ax.view_init(azim=camera_positions[1][0],  elev=camera_positions[1][1])
# ax.legend(groups, cl, loc=4)
fig.subplots_adjust(right=0.8)
lgnd = ax.legend(groups, cl.astype(int), loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=14, markerscale=30)
for handle in lgnd.legendHandles:
    handle.set_sizes([30.0])
fig.tight_layout()
#fig.set_rasterized(True)
plt.savefig( PLOTS + bl +  '_paper_DCAE.eps', dpi= dpi, format='eps')
plt.show()

##################################################################################################
#2D plots of UMAP and SAUCIE
z_UMAP  = DATA_ROOT + "Artificial_sets/UMAP_output/"
z_SAUCIE = DATA_ROOT + "Artificial_sets/SAUCIE_output/"

dpi = 350
rcParams['savefig.dpi'] = dpi
sz = 0.0001

fig = plt.figure(dpi=dpi, figsize=(11, 5))
# plot UMAP
# read data
infile = source_dir + 'set_' + str(bl) + '.npz'
npzfile = np.load(infile)
lbls = np.abs(npzfile['lbls'])
# read output
npz_res = np.load(z_UMAP + str(bl) + '_UMAP_rep_2D.npz')
z = npz_res['z']
smpl = np.random.choice(range(z.shape[0]), size=10000, replace=False)
lb = lbls[smpl]
z = z[smpl,:]
cl = np.unique(lb)

ax = fig.add_subplot(1, 2, 1)
ax.title.set_text('UMAP')
loc = plticker.MultipleLocator(base=5)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
colors = sns.color_palette("husl", n_colors=len(cl))
groups = []
for i in range(len(cl)):
    groups.append(
        ax.scatter(x=z[:, 0][lb == cl[i]], y=z[:, 1][lb == cl[i]], c=colors[i], s=sz))
    # ax.legend()
# plot SAUCIE
infile = source_dir + 'set_' + str(bl) + '.npz'
npzfile = np.load(infile)
lbls = np.abs(npzfile['lbls'])
# read output
npz_res = np.load(z_SAUCIE + str(bl) + '_SAUCIE_rep_2D.npz')
z = npz_res['z']
smpl = np.random.choice(range(z.shape[0]), size=10000, replace=False)
lb = lbls[smpl]
z = z[smpl,:]
cl = np.unique(lb)

ax = fig.add_subplot(1, 2, 2)
ax.title.set_text('SAUCIE')
loc = plticker.MultipleLocator(base=1)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
colors = sns.color_palette("husl", n_colors=len(cl))
groups = []
for i in range(len(cl)):
    groups.append(
        ax.scatter(x=z[:, 0][lb == cl[i]], y=z[:, 1][lb == cl[i]], c=colors[i], s=sz))
plt.savefig(PLOTS + bl + 'UMAP_SAUCIE_paper_DCAE.eps', dpi=dpi, format='eps')
        # ax.legend()


