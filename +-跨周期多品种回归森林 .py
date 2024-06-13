import fc
import pandas as pd,numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
test=1
if test==1:
    print('test');
y0=np.loadtxt('./data/multirb/y.txt', delimiter=",")
X0= np.array(eval(list(open('./data/multirb/X.txt'))[0]))[:,:-1].T
X0_diff=np.diff(X0,axis=0)
y0=y0[1:]
X0=np.concatenate((X0[1:],X0_diff),axis=1)
#X0=X0_diff
#生成特征X
y_labels=fc.sig(y0)
####
X_orig=X0
s=0
def main(n=2):
    global y_pred,y_test
    train_idx, test_idx = train_test_split(np.arange(X0.shape[0]), test_size=0.2, shuffle=True)
    if test==1:
        train_idx, test_idx =np.arange(len(train_idx)),len(train_idx)+np.arange(len(test_idx));
    X_train = X_orig[train_idx]
    X_test = X_orig[test_idx]
    y_train = y0[train_idx]#只有-1，0，1
    y_test = y_labels[test_idx]#只有-1，0，1
    y0_test = y0[test_idx]
    ####
    '更宽的阈值，且训练数据按照权重'
    y_train=y0[train_idx]
    n=3
    y_train_abs=1/(1+np.e**( (-abs(y_train)+n)*6 ))*(abs(y_train)+n)**0.2
    y_train_abs=np.where(y_train_abs <= 0.1, 0,y_train_abs)
    y_train=fc.sig(y_train)*y_train_abs
    #训练数据去掉0分类
    X_train=X_train[y_train!=0]
    y_train=y_train[y_train!=0]
    global s;s+=1
    print(len(X_train)) if s==1 else None
    print(len(y_test )) if s==1 else None
    ####
    rf_reg = RandomForestRegressor(max_features=1,
                                            #max_depth=5,
                                            n_estimators=10000,
                                           min_weight_fraction_leaf=0.05,
                                            #class_weight='balanced_subsample',
                                            #class_weight={0: 0.1, 1: 1-p_n, -1: p_n},
                                            #criterion='entropy',
                                           #random_state=42
                                           )
    rf_reg.fit(X_train,y_train)
    # 预测结果fc.sig(y0)*((abs(y0)-Thold)>=0)
    y_pred = rf_reg.predict(X_test)
    y_pred =fc.sig(y_pred)
    # 输出模型评估指标
    '''print(f'R^2:{rf_reg.score(X_test,y_test)}')
    print(f'相关性系数为：{np.corrcoef(y_pred,y_test)[0, 1]}')
    print('F1 score:', f1_score(y_test, y_pred, labels=[1,-1],average= 'weighted'))
    print('accuracy_score:',accuracy_score(y_test, y_pred))
    fc.plot([y_test,y_pred],k=1)
    fc.plot(fc.get_right_diff(y0_test,y_pred,cost=0))'''
    return f1_score(y_test, y_pred, labels=[1,-1],average= 'weighted'),sum(y0_test *y_pred)/len(y0_test)
for n in [0]:
    k=[];ri=[];print(n)
    for i in range(100):
        a=main(n)
        k+=[a[0]];ri+=[a[1]]
    print(f'f1:{np.mean(k)}')
    print(f'right:{np.mean(ri)}')
'''2.0
f1:0.4727188141105084
right:0.1716669503622927'''
