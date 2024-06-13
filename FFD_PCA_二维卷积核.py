import pandas as pd,numpy as np,time
from numpy.lib.stride_tricks import as_strided
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import fc
file_list='.\data\c_min.csv'
#file_list='.\data\c_day.csv'
df = pd.read_csv(file_list)
def main(step=420,
    n=425,
    hight=67,
    hight_y=50):
    c=df['close_Rollover'].to_numpy()
    c=np.log(c)
    core_numbers=16#核的数量
    PCA_components=32
    # 计算最终二维array的行数
    '''step=420
    n=425
    hight=67
    hight_y=50'''
    # 使用reshape函数将arr转换为一个二维array
    s0=np.array([c[i:i+step] for i in range(0, len(c)-step, n)])
    y0=np.array([c[i+step]-c[i+step-1] for i in range(0, len(c)-step, n)])
    s0=s0.copy()
    #one_hot = np.eye(np.max(arr)+1)[arr]
    for i,k in enumerate(s0):
        s0[i]=hight*(k-k[-1])/max((max(k)-k[-1]),(k[-1]-min(k)))+hight
    s0=s0.astype(int)
    s0_1hot=np.zeros((len(s0),int((max(np.max(k) for k in s0)+1)*len(s0[0]))))
    for i,k in enumerate(s0):
        s0_1hot[i]=(np.eye(s0_1hot.shape[1]//len(k))[k]).flatten()
    #print(s0_1hot.shape)
    # 将样本集s0转换为一个317 x step的矩阵
    X = np.array(s0_1hot)
    y=(y0/(max(y0)-min(y0))*hight_y//1).astype(int)
    #X, X_test, y, y_test= X[:-len(y)//4],X[-len(y)//4+3:],y[:-len(y)//4],y[-len(y)//4+3:]
    X, X_test, y, y_test =train_test_split(X, y, test_size=0.2)
    # 创建一个PCA对象并指定要压缩到的维度数
    pca = PCA(n_components=PCA_components)
    # 使用PCA对象对数据进行拟合和转换
    X_pca = pca.fit_transform(X)
    X_test_pca = pca.transform(X_test)
    # 绘制累计方差贡献率的曲线
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Components')
    plt.show()
    # 查看转换后的数据形状
    #print(X_pca.shape)
    compression_vectors = pca.components_
    #fc.plot(compression_vectors,k=1)
    ####
    # 将相关性矩阵作为输入进行k-means聚类
    #kmeans = KMeans(n_clusters=core_numbers, random_state=0).fit(X_pca)
    # 计算样本集合s0中所有样本的相关性矩阵
    corr_matrix_f = np.corrcoef(X_pca);#print(corr_matrix_f .shape)
    kmeans = KMeans(n_clusters=core_numbers, random_state=0).fit(corr_matrix_f)
    labels = kmeans.labels_
    ##
    la=[]
    for i in range(core_numbers):
        indices = np.where(kmeans.labels_ == i)[0]
        la +=[indices]
    re=[]
    for i in la:
        re+=[np.mean(s0[i], axis=0)]
    #fc.plot(re,k=1)
    matrix = np.reshape(compression_vectors, (X_pca.shape[1],step,s0_1hot.shape[1]//len(s0[0])))
    ####
    for i in matrix:
        img=[]
        if i.shape[1]>i.shape[0]:
            for k in i.T:
                img+=[[]]
                for m in k:
                    img[-1]+=[m]*(i.shape[1]//i.shape[0])
        elif i.shape[1]<i.shape[0]:
            for k in i.T:
                img+=[k.tolist()]*(i.shape[0]//i.shape[1])
        else:
            img=i.T;break
        #plt.imshow(img, cmap='gray');plt.show()
    ###随机森林
    X,y=X_pca,y
    X_test=X_test_pca
    from sklearn.ensemble import RandomForestRegressor
    #X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    # 创建一个随机森林回归模型
    rf_reg = RandomForestRegressor(max_features=1,
                                    #max_depth=5,
                                    n_estimators=1000,
                                    min_weight_fraction_leaf=0.05,
                                    #class_weight='balanced_subsample',
                                    #class_weight={0: 0.1, 1: 1-p_n, -1: p_n},
                                    #criterion='entropy',
                                   #random_state=42
                                   )

    # 训练模型
    rf_reg.fit(X,y)

    # 预测结果
    y_pred = rf_reg.predict(X_test)

    # 输出模型评估指标
    #print("Training set score: {:.2f}".format(rf_reg.score(X_test, y_test)))
    #print(f'相关性系数为：{np.corrcoef(y_pred,y_test)[0, 1]}')
    #fc.plot([y_test,y_pred],k=1)
    return np.corrcoef(y_pred,y_test)[0, 1]
