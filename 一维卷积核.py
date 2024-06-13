import pandas as pd,numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import fc
file_list='.\data\c_day.csv'
df = pd.read_csv(file_list)
c=df['close_Rollover'].to_numpy()
# 计算最终二维array的行数
num_rows = (len(c)- 22) // 10 + 1
# 使用reshape函数将arr转换为一个二维array
s0= as_strided(c, shape=(num_rows, 22), strides=(c.itemsize*10, c.itemsize))
s0=s0.copy()
for i,k in enumerate(s0):
    s0[i]=(k-np.mean(k))/np.std(k)
# 计算样本集合s0中所有样本的相关性矩阵
corr_matrix = np.corrcoef(s0)
print(corr_matrix.shape)
# 将相关性矩阵作为输入进行k-means聚类
kmeans = KMeans(n_clusters=16)
kmeans.fit(corr_matrix)
##
la=[]
for i in range(16):
    indices = np.where(kmeans.labels_ == i)[0]
    la +=[indices]
re=[]
for i in la:
    re+=[s0[i[0]]]
fc.plot(re,k=1,zoom='auto')
print(la)
####
# 将样本集s0转换为一个317 x 22的矩阵
X = np.array(s0)
# 创建一个PCA对象并指定要压缩到的维度数
pca = PCA(n_components=8)
# 使用PCA对象对数据进行拟合和转换
X_pca = pca.fit_transform(X)
# 查看转换后的数据形状
print(X_pca.shape)
compression_vectors = pca.components_
fc.plot(compression_vectors,k=1)
####
# 计算样本集合s0中所有样本的相关性矩阵
corr_matrix_f = np.corrcoef(X_pca)
print(corr_matrix_f .shape)
# 将相关性矩阵作为输入进行k-means聚类
kmeans = KMeans(n_clusters=16, random_state=0).fit(X_pca)
#kmeans = KMeans(n_clusters=16, random_state=0).fit(corr_matrix_f)
labels = kmeans.labels_
##
la=[]
for i in range(16):
    indices = np.where(kmeans.labels_ == i)[0]
    la +=[indices]
re=[]
for i in la:
    re+=[np.mean(s0[i], axis=0)]
fc.plot(re,k=1,zoom='auto')


