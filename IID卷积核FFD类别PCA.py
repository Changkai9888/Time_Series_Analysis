import pandas as pd,numpy as np,time
from numpy.lib.stride_tricks import as_strided
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import fc
file_list='.\data\c_day.csv'
#file_list='.\data\c_min.csv'
df = pd.read_csv(file_list)
def main(step=22,
    n=23,
    hight=50,#x的价格区间分辨率
    hight_y=50,#y(标签)的价格区间分辨率
    ):
    c=df['close_Rollover'].to_numpy()
    c=np.log(c)
    #core_numbers=32#核的数量
    # 计算最终二维array的行数
    '''step=22
    n=23
    hight=50#x的价格区间分辨率
    hight_y=50#y(标签)的价格区间分辨率'''
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
    #X, X_test, y, y_test= X[:-len(y0)//4],X[-len(y0)//4+1:],y0[:-len(y0)//4],y0[-len(y0)//4+1:]
    X, X_test, y, y_test =train_test_split(X, y0, test_size=0.2)
    y_lable=((y-min(y0))/(max(y0)-min(y0))*hight_y//1).astype(int)
    y=(y/(max(y0)-min(y0))*hight_y//1).astype(int)
    y_test=(y_test/(max(y0)-min(y0))*hight_y//1).astype(int)
    lable_num=[]
    for i in y_lable:
        if not i in lable_num:
            lable_num+=[i]
    #先做PCA再做线性回归，最后得到特征和卷积核。
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # 创建LDA模型，将样本从4维降至32维
    lda = LinearDiscriminantAnalysis(n_components=len(lable_num)-1)
    X_lda = lda.fit_transform(X, y_lable)
    X_test_lda = lda.transform(X_test)
    W = lda.scalings_
    #W= W[:, :core_numbers]
    #print("W shape: ", W.shape)
    ####
    matrix = np.reshape(W, (X_lda.shape[1],step,s0_1hot.shape[1]//len(s0[0])))
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
    X,y=X_lda,y
    X_test=X_test_lda
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    # 创建一个样本数据集
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

