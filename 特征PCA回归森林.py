import fc
import pandas as pd,numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
df_X=df.iloc[:, 2:]
X_orig=df_X.values
X_norm=fc.normal_2D(X_orig)
pca = PCA(n_components=0.95)
pca.fit(X_norm)
X_pca = pca.transform(X_orig)
Volat=df['Volat'].values
Volat=30.572*Volat-16.798*Volat**2
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.show()
compression_vectors = pca.components_
print(len(compression_vectors))
#print(compression_vectors)
#####
Volat=Volat[:-1];
X_norm=X_norm[:-1];
X_pca=X_pca[:-1]
X_orig=X_orig[:-1]
def main():
    close=df ['C_Rollover'].to_numpy()
    y=np.diff(close)
    close=close[:-1];
    #y[abs(y) <Volat ]=0;  #y[y >Volat ]=1; y[y <-Volat ]=-1;
    X, X_test, y, y_test =train_test_split(X_pca, y, test_size=0.2)
    #X, X_test, y, y_test= X_pca[:-len(y)//4],X_pca[-len(y)//4+3:],y[:-len(y)//4],y[-len(y)//4+3:]
    rf_reg = RandomForestRegressor(max_features=1,
                                            #max_depth=5,
                                            n_estimators=100,
                                           min_weight_fraction_leaf=0.05,
                                            #class_weight='balanced_subsample',
                                            #class_weight={0: 0.1, 1: 1-p_n, -1: p_n},
                                            #criterion='entropy',
                                           #random_state=42
                                           )
    rf_reg.fit(X,y)
    # 预测结果
    y_pred = rf_reg.predict(X_test)
    # 输出模型评估指标
    #print("Training set score: {:.4f}".format(rf_reg.score(X_test,y_test)))
    #print(f'相关性系数为：{np.corrcoef(y_pred,y_test)[0, 1]}')
    #fc.plot([y_test,y_pred],k=1)
    return rf_reg.score(X_test,y_test),sum(y_test *y_pred)/len(y_test)
k=[];ri=[]
for i in range(100):
    k+=[main()[0]];ri+=[main()[1]]
print(f'f1:{np.mean(k)}')
print(f'right:{np.mean(ri)}')
