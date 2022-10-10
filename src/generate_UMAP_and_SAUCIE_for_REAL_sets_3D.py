'''
Generates UMAP and SAUCIE for real data sets in 3D
run with tensorflow v 1.12, python 3.6.0 (to satisfy SAUCIE requirements)
python -m pip install --upgrade ~/Downloads/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
'''

import numpy as np
import umap.umap_ as umap
from plotly.io import to_html
from plotly.graph_objs import Scatter3d
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
import seaborn as sns
from matplotlib.colors import rgb2hex


def plot3D_cluster_colors(z, lbls, camera=None, legend=True, msize=1):
    x = z[:, 0]
    y = z[:, 1]
    z1 = z[:, 2]

    lbls_list = np.unique(lbls)
    nM = len(np.unique(lbls))

    palette = sns.color_palette("husl", nM)
    colors = np.array([rgb2hex(palette[i]) for i in range(len(palette))])

    fig = go.Figure()
    for m in range(nM):
        IDX = [x == lbls_list[m] for x in lbls]
        xs = x[IDX];
        ys = y[IDX];
        zs = z1[IDX];
        fig.add_trace(Scatter3d(x=xs, y=ys, z=zs,
                                name=lbls_list[m],
                                mode='markers',
                                marker=dict(
                                    size=msize,
                                    color=colors[m],
                                    opacity=0.5,
                                ),
                                text=lbls[IDX],
                                hoverinfo='text'))
        fig.update_layout(yaxis=dict(range=[-3, 3]),
                          margin=dict(l=0, r=0, b=0, t=10))
        if camera == None:
            camera = dict(
                up=dict(x=0, y=0., z=1),
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        fig.update_layout(scene_camera=camera, showlegend=legend)
        # set colour to white
        fig.update_layout(dict(xaxis=dict( showgrid=True, gridwidth=1, gridcolor="#eee")))
        fig.update_layout(scene=dict(
            xaxis=dict(visible=True, backgroundcolor='rgba(0,0,0,0)', showgrid=True, gridcolor="#eee", gridwidth=1,
                       showline=True, zeroline=True),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', showgrid=True, gridcolor="#eee", gridwidth=1, showline=True,
                       zeroline=True),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', showgrid=True, gridcolor="#eee", gridwidth=1, showline=True,
                       zeroline=True)
        ))

    fig.update_layout(legend=dict(font=dict(family="Courier", size=20, color="black"),itemsizing='constant' ),
                      legend_title=dict(font=dict(family="Courier", size=30, color="blue" )) )
    return fig

DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'CyTOFdataPreprocess/'
output_dir  = DATA_ROOT + 'Real_sets/UMAP_output/'


list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz']

for bl in list_of_inputs:
    infile = source_dir + bl
    npzfile = np.load(infile,  allow_pickle=True)
    aFrame = npzfile['aFrame'];
    lbls= npzfile['lbls']
    mapper = umap.UMAP(n_neighbors=30, n_components=3, metric='euclidean', random_state=42, min_dist=0, low_memory=True).fit(aFrame)
    yUMAP =  mapper.transform(aFrame)

    fig = plot3D_cluster_colors(yUMAP, lbls=lbls)
    html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                       include_mathjax=False, post_script=None, full_html=True,
                       animation_opts=None, default_width='100%', default_height='100%', validate=True)
    html_dir = output_dir
    Html_file = open(html_dir + "/" + str(bl) + 'UMAP' + '_' + "_Buttons_3D.html", "w")
    Html_file.write(html_str)
    Html_file.close()

    np.savez(output_dir + '/' + str(bl) + '_UMAP_rep_3D.npz', z=yUMAP)


#SAUCIE
# one need to install tf.12 and swithch python env to run this part of code
output_dir  = DATA_ROOT + 'Real_sets/SAUCIE_output/'
import sys
sys.path.append("/media/grinek/Seagate/DCAE/SAUCIE")
sys.path.append("/media/grinek/Seagate/DCAE")
import SAUCIE
import tensorflow as tf

for bl in list_of_inputs:
    infile =  source_dir + bl
    npzfile = np.load(infile,  allow_pickle=True)

    aFrame = npzfile['aFrame'];
    lbls= npzfile['lbls']

    tf.reset_default_graph()
    saucie = SAUCIE.SAUCIE(aFrame.shape[1], layers=[512,256,128,3])
    loadtrain = SAUCIE.Loader(aFrame, shuffle=True)
    saucie.train(loadtrain, steps=10000)

    loadeval = SAUCIE.Loader(aFrame, shuffle=False)
    ySAUCIE = saucie.get_embedding(loadeval)

    fig = plot3D_cluster_colors(ySAUCIE, lbls=lbls)
    html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                       include_mathjax=False, post_script=None, full_html=True,
                       animation_opts=None, default_width='100%', default_height='100%', validate=True)
    html_dir = output_dir
    Html_file = open(html_dir + "/" + str(bl) + '_SAUCIE' + '_' + "_Buttons_3D.html", "w")
    Html_file.write(html_str)
    Html_file.close()

    np.savez(output_dir + '/' + str(bl) + '_SAUCIE_rep_3D.npz', z=ySAUCIE)
