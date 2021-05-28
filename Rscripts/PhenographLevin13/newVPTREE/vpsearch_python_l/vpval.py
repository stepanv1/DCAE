import vpsearch as vp
import numpy as np
import time
import pandas as pd

df = pd.read_csv("Samusik.csv")
channels = ["Y89_CD45","Cd112_CD45RA","In115_IgM","La139_CD270","Pr141_CD49D","Nd142_CD19","Nd143_HLA_ABC","Nd144_CD72","Nd145_CXCR5","Nd146_CD48","Sm147_CD20","Nd148_CD274","Sm149_CD200","Nd150_CD43","Sm152_CD21","Eu153_CD124","Sm154_CD84","Gd155_IgG","Gd156_CD10","Gd157_HLA_E","Gd158_CD194","Tb159_CD22","Dy161_CD80","Dy162_CD79B","Dy163_CD23","Dy164_CD86","Ho165_CD40","Er166_CD24","Er167_CD27","Er168_CD30","Tm169_CD25","Yb171_CD44","Yb172_CD38","Yb173_BTLA","Yb174_HLA_DR","Lu175_CD184","Yb176_CD83"]
df_2 = pd.read_csv("master.csv")


df_np = df_2[channels].as_matrix()
sample = df
data = sample.as_matrix()
start = time.time()
indVP, distVP  = vp.find_nearest_neighbors(df_np, 30, 1)
end = time.time()
print("Time\n")
print(end - start)

start = time.time()
indVP, distVP  = vp.find_nearest_neighbors(df_np, 30, 4)
end = time.time()
print("Time\n")
print(end - start)
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
