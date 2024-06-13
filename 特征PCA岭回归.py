import fc
import pandas as pd,numpy as np,time
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
p=['Datetime', 'C_Rollover', 'C_normal', 'Spread_normal', 'CUSUM_0', 'CUSUM_8',\
   'Slope', 'LR_Std', 'LR_Value', 'LR_Value_normal', 'Volat', 'X_std', 'Skew', 'Kurt', 'Imbal_IN', \
   'Imbal_OUT', 'Coeff_cv', 'hour_name', 'week_name', 'Mod_c', 'Oneside_Last', 'Peace_Last', \
   'Boom_Last', 'Highest_past', 'Lowest_past']
file_list='.\data\c_h_all.csv'#c_day_cusum_c.csv'#
df = pd.read_csv(file_list)
df ['CUSUM_0-8']= df.iloc[:, 4] - df.iloc[:, 5]
df=pd.concat([df, df.iloc[:, 2:].diff().add_suffix('_diff')], axis=1)[1:]
#生成特征X
df_X=df.iloc[:, 2:]
X_orig=df_X.values
X_norm=fc.normal_2D(X_orig)
pca = PCA(n_components=0.95)
pca.fit(X_norm)
X_pca = pca.transform(X_orig)
Volat=df['Volat'].values
Volat=12.9312658*Volat-9.21834318*Volat**2
'''plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.show()'''
compression_vectors = pca.components_
print(len(compression_vectors))
#print(compression_vectors)
#####
close=df ['C_Rollover'].to_numpy()
y0=np.diff(close);
y=np.diff(close);
Volat=Volat[:-1];
#y[abs(y) <Volat ]=0; y[y >Volat ]=1; y[y <-Volat ]=-1; 
X_norm=X_norm[:-1];
X_pca=X_pca[:-1]
X_orig=X_orig[:-1]
close=close[:-1];
def main():
    #X_train, X_test, y_train, y_test =train_test_split(X_pca, y, test_size=0.2)
    train_idx, test_idx = train_test_split(np.arange(X_pca.shape[0]), test_size=0.2, shuffle=True)
    X_train = X_pca[train_idx]
    X_test = X_pca[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    y0_test = y0[test_idx]
    #X_train, X_test, y_train, y_test= X_pca[:-len(y)//4],X_pca[-len(y)//4+3:],y[:-len(y)//4],y[-len(y)//4+3:]

    # 进行岭回归
    #ridge = Ridge(alpha=0.1)
    ridge = LinearRegression()
    ridge.fit(X_train,y_train)
    # 预测结果并计算 R 平方值
    y_pred  = ridge.predict(X_test)
    r2 = r2_score(y_test,y_pred )
    #fc.plot([y_test,y_pred],k=1)
    return r2,np.corrcoef(y_pred,y_test)[0, 1],ridge.coef_
###
ti0=time.time()
k0=[];k=[]
for i in range(1000):
    k0+=[main()[0]]
    k+=[main()[1]]
print(np.mean(k0,axis=0));print(np.std(k0,axis=0))
print(np.mean(k,axis=0));print(np.std(k,axis=0))
print(f'用时：{time.time()-ti0}s')

#[12.9312658  -9.21834318]
