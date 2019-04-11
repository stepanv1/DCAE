import vpsearch as vp
import numpy as np
import time
import pandas as pd
import hnswlib
from scipy.ndimage.interpolation import shift
import hnswlib
from scipy.ndimage.interpolation import shift
def find_neighbors(data, k_, metric='euclidean', cores=1):
   dim = np.shape(data)[1]
   num_elements = np.shape(data)[0]
   data_labels = np.arange(num_elements)
   p = hnswlib.Index(space='l2', dim=dim)
   p.init_index(max_elements=num_elements, ef_construction=5000, M=16)
   # Element insertion (can be called several times):
   p.add_items(data, data_labels, num_threads=cores)
   # Controlling the recall by setting ef:
   p.set_ef(5000)  # ef should always be > k
   # Query dataset, k - number of closest elements (returns 2 numpy arrays)
   labels, distances = p.knn_query(data, k=k_+1,num_threads=cores)
   #correct results
   IDX = distances[:,0]!=0
   distances = np.array([shift(distances[i,:], 1, cval=0) if IDX[i] else distances[i,:] for i in data_labels ])
   labels = np.array([shift(labels[i, :], 1, cval=0) if IDX[i] else labels[i, :] for i in data_labels])
   return {'dist':distances[:,1:], 'idx':labels[:,1:]}
df = pd.read_csv("Samusik.csv")
channels = ["Y89_CD45","Cd112_CD45RA","In115_IgM","La139_CD270","Pr141_CD49D","Nd142_CD19","Nd143_HLA_ABC","Nd144_CD72","Nd145_CXCR5","Nd146_CD48","Sm147_CD20","Nd148_CD274","Sm149_CD200","Nd150_CD43","Sm152_CD21","Eu153_CD124","Sm154_CD84","Gd155_IgG","Gd156_CD10","Gd157_HLA_E","Gd158_CD194","Tb159_CD22","Dy161_CD80","Dy162_CD79B","Dy163_CD23","Dy164_CD86","Ho165_CD40","Er166_CD24","Er167_CD27","Er168_CD30","Tm169_CD25","Yb171_CD44","Yb172_CD38","Yb173_BTLA","Yb174_HLA_DR","Lu175_CD184","Yb176_CD83"]
df_2 = pd.read_csv("master.csv")


df_np = df_2[channels].as_matrix()
sample = df
data = sample.as_matrix()
print("Start\n")
start = time.time()
find_neighbors(df_np, 30, metric='euclidean', cores=4)
end = time.time()
print("Time\n")
print(end - start)

# start = time.time()
# indVP, distVP  = vp.find_nearest_neighbors(df_np, 30, 4)
# end = time.time()
# print("Time\n")
# print(end - start)
# start = time.time()
# indVP, distVP  = vp.find_nearest_neighbors(data, 30, 10)
# end = time.time()
# print("Time\n")
# print(end - start)
#
# #
#indVP2, distVP2  = vp.find_nearest_neighbors(data, 30, 1)
# from  sklearn.neighbors import BallTree
# tree = BallTree(data, leaf_size=40,  metric="euclidean")
# dist, ind = tree.query(data, k=31)
#
# print(sum(ind[:,1:31]!=indVP))
# print(sum(indVP!=ind[:,1:31]))
# print(sum(np.where(ind[:,1:31]!=indVP[:100,:])))
# print("Finished")
#
# for i in range(0,100):
#     test = ind[i][1:32]==indVP[i]
#     if False in test:
#         print(i)
#         print ind[i][1:32]
#         print indVP[i]
#         print ("------------")
