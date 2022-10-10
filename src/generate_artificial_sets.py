'''
Generates artificial clusters and computes objects required to run
DCAE
generate UMAP data and plots for comparison
'''

import numpy as np
import os


from utils_evaluation import  preprocess_artificial_clusters,  generate_clusters_pentagon

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