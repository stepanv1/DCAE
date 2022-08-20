'''
Generates UMAP and SAUCIE
mappings as well as performance metrics
for real data sets
run with tensorflow v 1.12, python 3.6.0 (to satisfy SAUCIE requirements)
python -m pip install --upgrade ~/Downloads/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
'''

import numpy as np

import os

import umap.umap_ as umap
import pandas as pd
import timeit
from plotly.io import to_html
from plotly.graph_objs import Scatter
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
import seaborn as sns
from matplotlib.colors import rgb2hex
#from  utils_evaluation import  plot2D_cluster_colors
def plot2D_cluster_colors(z, lbls, legend=True, msize=1):
    x = z[:, 0]
    y = z[:, 1]

    lbls_list = np.unique(lbls)
    nM = len(np.unique(lbls))

    palette = sns.color_palette('husl', nM)
    colors = np.array([rgb2hex(palette[i]) for i in range(len(palette))])

    fig = go.Figure()
    for m in range(nM):
        IDX = [x == lbls_list[m] for x in lbls]
        xs = x[IDX];
        ys = y[IDX];
        fig.add_trace(Scatter(x=xs, y=ys,
                                name=lbls_list[m],
                                mode='markers',
                                marker=dict(
                                    size=msize,
                                    color=colors[m],  # set color to an array/list of desired values
                                    opacity=0.5,
                                ),
                                text=lbls[IDX],
                                # hoverinfo='text')], filename='tmp.html')
                                hoverinfo='text'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=10))
        fig.update_layout(showlegend=legend)
        fig.update_layout(dict(paper_bgcolor='white', plot_bgcolor='white'),
                          xaxis=dict(showgrid=True, gridcolor="#eee", gridwidth=1, showline=True, zeroline=True),
                          yaxis=dict(showgrid=True, gridcolor="#eee", gridwidth=1, showline=True, zeroline=True)
                          )
    fig.update_layout(legend=dict(font=dict(family="Courier", size=20, color="black"), itemsizing='constant'),
                      legend_title=dict(font=dict(family="Courier", size=30, color="blue")))
    return fig


DATA_ROOT = '/media/grinek/Seagate/'

#UMAP
source_dir = DATA_ROOT + 'CyTOFdataPreprocess/'
output_dir  = DATA_ROOT + 'Real_sets/UMAP_output/'


list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz']
#bl = list_of_inputs[0]
for bl in list_of_inputs:
    infile = source_dir + bl
    # markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
    # np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
    npzfile = np.load(infile,  allow_pickle=True)
    # = weight_distALL[IDX,:]
    aFrame = npzfile['aFrame'];
    lbls= npzfile['lbls']
    mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=True).fit(aFrame)
    yUMAP =  mapper.transform(aFrame)

    fig = plot2D_cluster_colors(yUMAP, lbls=lbls)
    html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                       include_mathjax=False, post_script=None, full_html=True,
                       animation_opts=None, default_width='100%', default_height='100%', validate=True)
    html_dir = output_dir
    Html_file = open(html_dir + "/" + str(bl) + 'UMAP' + '_' + "_Buttons.html", "w")
    Html_file.write(html_str)
    Html_file.close()

    np.savez(output_dir + '/' + str(bl) + '_UMAP_rep_2D.npz', z=yUMAP)


#SAUCIE
# one need to install tf.12 and swithc python env to run  this part of code
output_dir  = DATA_ROOT + 'Real_sets/SAUCIE_output/'
import sys
sys.path.append("/media/grinek/Seagate/DCAE/SAUCIE")
sys.path.append("/media/grinek/Seagate/DCAE")
import SAUCIE
#from importlib import reload
#bl = list_of_branches[1]
#tf.compat.v1.disable_eager_execution()
import tensorflow as tf
#from keras import backend.clear_session
#saucie = SAUCIE.SAUCIE(30)
for bl in list_of_inputs:
    infile =  source_dir + bl    # markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
    # np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
    npzfile = np.load(infile,  allow_pickle=True)
    # = weight_distALL[IDX,:]
    aFrame = npzfile['aFrame'];
    lbls= npzfile['lbls']


    tf.reset_default_graph()
    saucie = SAUCIE.SAUCIE(aFrame.shape[1])
    loadtrain = SAUCIE.Loader(aFrame, shuffle=True)
    saucie.train(loadtrain, steps=10000)

    loadeval = SAUCIE.Loader(aFrame, shuffle=False)
    ySAUCIE = saucie.get_embedding(loadeval)
    # np.savez('LEVINE32_' + 'embedSAUCIE_100000.npz', embedding=embedding)

    fig = plot2D_cluster_colors(ySAUCIE, lbls=lbls)
    html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                       include_mathjax=False, post_script=None, full_html=True,
                       animation_opts=None, default_width='100%', default_height='100%', validate=True)
    html_dir = output_dir
    Html_file = open(html_dir + "/" + str(bl) + '_SAUCIE' + '_' + "_Buttons.html", "w")
    Html_file.write(html_str)
    Html_file.close()

    np.savez(output_dir + '/' + str(bl) + '_SAUCIE_rep_2D.npz', z=ySAUCIE)
