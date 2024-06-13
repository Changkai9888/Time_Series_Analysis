import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import f1_score
import fc
from sklearn.linear_model import Ridge
from data.multirb import 数据格式处理
test=0
if test==1:#0随机；1，连续
    print('test');
X0,y0=数据格式处理.get_data()
y0=np.diff(y0)*2000
#X0=np.concatenate((X0[:,:2],X0[:,-1][:,np.newaxis]),axis=1)#X0截取
X0=X0[:-1]
X0_diff=np.diff(X0,axis=0)
#X0_diff_2=np.diff(X0,n=2,axis=0)
y0=y0[1:]
X0=X0[1:]
#X0=np.concatenate((X0,X0_diff,(X0[:,0]*X0[:,1])[:,np.newaxis]),axis=1)
X0=np.concatenate((X0,X0_diff),axis=1)
#X0=X0_diff
print(X0.shape[0]==y0.shape[0])
#X0=X0_diff
#X0=(X0.T).tolist()
#X0=np.array(X0[0:3]+X0[4:7]+X0[8:11]+X0[12:15]+X0[16:19]+X0[20:23]).T
#X0=np.array([X0[0],X0[4],X0[8],X0[12],X0[16],X0[20]]).T
####
s=0
print(X0.shape)
def main(func = LinearRegression()):
    global y_pred,y_test,train_idx,test_idx,X_train,y_train
    #随机
    train_idx, test_idx = train_test_split(np.arange(X0.shape[0]), test_size=0.2, shuffle=True)
    if test==1:#连续
        train_idx, test_idx =np.arange(len(train_idx)),len(train_idx)+np.arange(1,len(test_idx));
    #np.random.shuffle(train_idx);np.random.shuffle(test_idx);
    X_train=X0[train_idx]
    X_test = X0[test_idx]
    y_train = y0[train_idx]
    y_test =  y0[test_idx]
    ####
    '更宽的阈值，且训练数据按照权重'
    y_train=y0[train_idx]
    n=2
    y_train_abs=1/(1+np.e**( (-abs(y_train)+n)*6 ))*(abs(y_train)+n)**0.2
    y_train_abs=np.where(y_train_abs <= 0.1, 0,y_train_abs)
    y_train=fc.sig(y_train)*y_train_abs
    y_train=fc.sig(y_train)
    #训练数据去掉0分类
    global s;s+=1
    print(f'去掉0分类{len(X_train[y_train==0])}个') if s==1 else None
    X_train=X_train[y_train!=0]
    y_train=y_train[y_train!=0]
    ####
    print(len(X_train)) if s==1 else None
    print(len(y_test )) if s==1 else None
    ####
    func.fit(X_train,y_train)
    # 预测结果并计算 R 平方值
    y_pred  = func.predict(X_test)
    r2 = r2_score(y_test,y_pred )
    #fc.plot([y_test,y_pred],k=1)
    '+-特征回归'
    global right_list
    right_list=sum(y_test * fc.sig(y_pred))/len(y_test)-max(0,np.mean(y_test)*np.mean(y_pred))
    right_std=np.std(y_test * fc.sig(y_pred))/len(y_test)**0.5
    return f1_score(fc.sig(y_test), fc.sig(y_pred), labels=[1,-1],average= 'weighted'),\
           right_list,\
           r2,right_std
###
def train(func ):
    f1=[];r2=[];right=[];vector=[];std=[]
    for i in range(1000):
        result=main(func )
        f1+=[result[0]];
        right+=[result[1]]
        vector+=[func.coef_]
        r2+=[result[2]]
        std+=[result[3]]
    if len(result)==4:
        print(f'f1:{np.mean(f1,axis=0)}');#print(np.std(k0,axis=0))
        print(f'right:{np.mean(right,axis=0)}');#print(np.std(k,axis=0))
        print(f'R^2:{np.mean(r2)}');#print(np.std(k,axis=0))
        print(f'std:{np.mean(std)}')
        return np.mean(vector,axis=0),np.std(vector,axis=0)
    return
print('Linear=============')
vect0=train(LinearRegression())#普通线性回归
#[12.9312658  -9.21834318]
#day[Thold=0.4647*ATR-0.0040295*ATR**2+21.77*Volat]
print('ElasticNet===========')
vect1=train(ElasticNet(alpha=0.2, l1_ratio=0.5))#时间序列平稳最小二乘法线性回归
#day[Thold=0.6439*ATR-0.006462*ATR**2+1.64845*Volat]
print('Ridge=============')
vect2=train(Ridge(alpha=0.1))
vect=np.array([vect0,vect1,vect2])
#for i in vect[2,0]:
    #print(f'{i},',end="")
fc.plot(vect[:,0],k=1)
