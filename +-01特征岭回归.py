import fc
import pandas as pd,numpy as np,time
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import f1_score
p=['Datetime', 'C_Rollover', 'Volat', 'C_in_HL', 'C_normal', 'Spread_normal',\
   'CUSUM_0', 'CUSUM_8', 'Slope', 'LR_Std', 'LR_Value', 'LR_Value_normal', \
   'X_std', 'Skew', 'Kurt', 'Imba_IN_normal', 'Imba_OUT_normal', 'Coeff_HV', \
   'Coeff_LV', 'time_self_name', 'time_uper_name', 'Mod_c', 'Oneside_Last',\
   'Peace_Last', 'Boom_Last', 'Highest_past', 'Lowest_past','ATR']
file_list='.\data\c_day_all.csv'#c_day_cusum_c.csv'#
df = pd.read_csv(file_list)
df ['CUSUM_0-8']= df['CUSUM_0'] - df['CUSUM_8']

df=pd.concat([df, df.iloc[:, 2:].diff().add_suffix('_diff')], axis=1)[1:]
#生成特征X
y0=np.diff(df ['C_Rollover'])
close=df ['C_Rollover'].to_numpy()[:-1];
####
df_X=df.iloc[:, 2:][:-1];
X_orig=df_X.values
####
time_self_name=df_X['time_self_name'].to_numpy().reshape(-1,1)
time_uper_name=df_X['time_uper_name'].to_numpy().reshape(-1,1)
Volat=df_X['Volat'].to_numpy().reshape(-1,1)
ATR=df_X['ATR'].to_numpy().reshape(-1,1)
Thold=np.concatenate((ATR,ATR**2,Volat,time_self_name**2), axis=1)
#X0=ATR[:-1,0].reshape(-1,1)#单ATR
X0=X_orig
#X0=Thold#01模式
y0=y0#+-模式
#y0=abs(y0)#01模式
'PCA'
#X_norm=fc.normal_2D(X0)
#pca = PCA(n_components=0.95)
#X0 = pca.fit_transform(X0)
print(X0.shape)
####
def main(func = LinearRegression()):
    global y_pred,y_test
    train_idx, test_idx = train_test_split(np.arange(X0.shape[0]), test_size=0.2, shuffle=True)
    #train_idx, test_idx =np.arange(len(train_idx)),len(train_idx)+np.arange(len(test_idx))
    X_train=X0[train_idx]
    X_test = X0[test_idx]
    y_train = y0[train_idx]
    y_test =  y0[test_idx]
    #X, X_test, y, y_test =train_test_split(X0, y0, test_size=0.2)
    ####
    func.fit(X_train,y_train)
    # 预测结果并计算 R 平方值
    y_pred  = func.predict(X_test)
    r2 = r2_score(y_test,y_pred )
    #fc.plot([y_test,y_pred],k=1)
    #f1_score(fc.sig(y_test), fc.sig(y_pred), labels=[1,-1],average= 'weighted')
    #return func.coef_
    '0-1特征回归'
    #return r2,np.corrcoef(y_test,y_pred)[0, 1]
    '+-特征回归'
    return f1_score(fc.sig(y_test), fc.sig(y_pred), labels=[1,-1],average= 'weighted'),\
           sum(y_test * fc.sig(y_pred))/len(y_test)-max(0,np.mean(y_test)*np.mean(y_pred)),\
           r2,func.coef_
###
def train(func ):
    k0=[];k=[];right=[];vector=[]
    for i in range(1000):
        result=main(func )
        k0+=[result[0]];k+=[result[1]]
        vector+=[func.coef_]
        if len(result)==4:
            right+=[result[2]]
    if len(result)==2:
        print(f'R^2:{np.mean(k0,axis=0)}');#print(np.std(k0,axis=0))
        print(f'corr:{np.mean(k,axis=0)}');#print(np.std(k,axis=0))
        return np.mean(vector,axis=0),np.std(vector,axis=0)
    if len(result)==4:
        print(f'f1:{np.mean(k0,axis=0)}');#print(np.std(k0,axis=0))
        print(f'right:{np.mean(k,axis=0)}');#print(np.std(k,axis=0))
        print(f'R^2:{np.mean(right)}');#print(np.std(k,axis=0))
        return np.mean(vector,axis=0),np.std(vector,axis=0)
    return
print('Linear')
vect=train(LinearRegression())#普通线性回归
#[12.9312658  -9.21834318]
#day[Thold=0.4647*ATR-0.0040295*ATR**2+21.77*Volat]
print('ElasticNet')
vect=train(ElasticNet(alpha=0.2, l1_ratio=0.5))#时间序列平稳最小二乘法线性回归
#day[Thold=0.6439*ATR-0.006462*ATR**2+1.64845*Volat]
print('Ridge')
vect=train(Ridge(alpha=0.1))
vect=np.array(vect)
print(vect[0])
