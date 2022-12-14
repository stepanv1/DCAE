'''
create 2D projections of DCAE, UMAP and SAUCIE outputs for the manuscript; single big plot for
real sets
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import rcParams
import numpy as np
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import rgb2hex
from utils_evaluation import table
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

ID = 'DCAE'+  '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)

os.chdir('/media/grinek/Seagate/DCAE/')
DATA_ROOT = '/media/grinek/Seagate/'
DATA_DIR = DATA_ROOT + 'CyTOFdataPreprocess/'
source_dir = DATA_ROOT + 'Real_sets/'
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
                  'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz', 'Shenkareuclid_shifted.npz', 'Samusik_01.npz']
PLOTS = DATA_ROOT + "Real_sets/PLOTS/"
z_dir = DATA_ROOT + 'Real_sets/DCAE_output/'
output_dir = DATA_ROOT + "Real_sets/DCAE_output/Performance/"
temp_dir  = DATA_ROOT + "Real_sets/DCAE_output/Performance/temp/"

bl_index  = [0,1,2,3]
#azymuth, elevaation , position
camera_positions = [[[46,10,0], [0,0,0], [0,0,0]], [[-98,-6,0], [0,0,0], [0,0,0]], [[-157,-54,0], [0,0,0], [0,0,0]],
                    [[-68,13,0], [0,0,0], [0,0,0]]]
epochs = 1000

idx = bl_index[1]
unassigned_lbls = ['"unassigned"', '"Unassgined"', '-1', '-1']
for idx in bl_index:
    print(output_dir)
    bl = list_of_inputs [idx]
    print(bl)
    # read data
    infile = DATA_DIR + bl
    npzfile = np.load(infile, allow_pickle=True)
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz', allow_pickle=True)
    z = npz_res['z']

    cl = np.unique(lbls)

    if idx==0:
        n_samples = 75000
        smpl = np.random.choice(range(z.shape[0]), size=n_samples, replace=False)
        lbls = lbls[smpl]
        z = z[smpl,:]
    if idx == 1:
        n_samples = 200000
        smpl = np.random.choice(range(z.shape[0]), size=n_samples, replace=False)
        lbls = lbls[smpl]
        z = z[smpl, :]

    dpi = 350
    rcParams['savefig.dpi'] = dpi
    sz=1
    fig = plt.figure(dpi = dpi, figsize=(10,10))
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

    nM = len(np.unique(lbls))
    palette = sns.color_palette("husl", nM)
    colors = np.array([rgb2hex(palette[i]) for i in range(len(palette))])

    groups = []
    for i in range(len(cl)):
        if cl[i] != unassigned_lbls[idx]:
            groups.append(ax.scatter(xs=z[:,0][lbls==cl[i]], ys=z[:,1][lbls==cl[i]], zs=z[:,2][lbls==cl[i]], c = colors[i],  s=sz, alpha=0.2))

    ax.view_init(azim=camera_positions[idx][0][0],  elev=camera_positions[idx][0][1])
    ax.set_rasterized(True)
    if idx==2:
       lgnd = ax.legend(groups, cl[cl != unassigned_lbls[idx]], loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=16, markerscale=30, ncol=2)
    else:
        lgnd = ax.legend(groups, cl[cl != unassigned_lbls[idx]], loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=16, markerscale=30)
    for handle in lgnd.legendHandles:
        handle.set_sizes([30.0])
    plt.savefig( PLOTS + list_of_inputs[idx] +  '_paper__single_DCAE.eps', dpi= dpi, format='eps', bbox_inches='tight')
    plt.savefig( PLOTS + list_of_inputs[idx] +  '_paper__single_DCAE.tif', dpi= dpi, format='tif', bbox_inches='tight')
