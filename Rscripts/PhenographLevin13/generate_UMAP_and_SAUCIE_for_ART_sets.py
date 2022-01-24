'''
Generates UMAP and SAUCIE
mappings for artificial clusters
run with tensorflow v 1.12, python 3.6.0 (to satisfy SAUCIE requirements)
python -m pip install --upgrade ~/Downloads/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
'''

import numpy as np

import os

import umap.umap_ as umap
import pandas as pd
import timeit
from plotly.io import to_html
import plotly.io as pio
pio.renderers.default = "browser"

from  utils_evaluation import plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, show3d


os.chdir('/home/grinek/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/grinek/Seagate/'

#UMAP
output_dir =  DATA_ROOT + "Artificial_sets/UMAP_output/"
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'


list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
#bl = list_of_branches[1]
for bl in list_of_branches:
    infile = source_dir + 'set_' + str(bl) + '.npz'
    # markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
    # np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
    npzfile = np.load(infile)
    # = weight_distALL[IDX,:]
    aFrame = npzfile['aFrame'];
    lbls= npzfile['lbls']
    mapper = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=True).fit(aFrame)
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
output_dir =  DATA_ROOT + "Artificial_sets/SAUCIE_output/"

import sys
sys.path.append("/home/grinek/PycharmProjects/BIOIBFO25L/SAUCIE")
sys.path.append("/home/grinek/PycharmProjects/BIOIBFO25L")
import SAUCIE
from importlib import reload
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
#bl = list_of_branches[1]
#tf.compat.v1.disable_eager_execution()
import tensorflow as tf
#from keras import backend.clear_session
#saucie = SAUCIE.SAUCIE(30)
for bl in list_of_branches:
    infile = source_dir + 'set_' + str(bl) + '.npz'
    # markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
    # np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
    npzfile = np.load(infile)
    # = weight_distALL[IDX,:]
    aFrame = npzfile['aFrame'];
    lbls= npzfile['lbls']


    tf.reset_default_graph()
    saucie = SAUCIE.SAUCIE(30)
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
