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
y_labels=fc.sig(y0)
####
df_X=df.iloc[:, 2:]
X_orig=df_X.values[:-1];
s=0
def main(n=2):
    train_idx, test_idx = train_test_split(np.arange(X_orig.shape[0]), test_size=0.2, shuffle=True)
    #train_idx, test_idx =np.arange(len(train_idx)),len(train_idx)+np.arange(len(test_idx))
    X_train = X_orig[train_idx]
    X_test = X_orig[test_idx]
    y_train = y_labels[train_idx]#只有-1，0，1
    y_test = y_labels[test_idx]#只有-1，0，1
    y0_test = y0[test_idx]
    ####
    '更宽的阈值，且训练数据按照权重'
    y_train=y0[train_idx]
    n=n
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
                                            n_estimators=100,
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
for n in np.arange(0,10,0.5):
    k=[];ri=[];print(n)
    for i in range(1000):
        a=main(n)
        k+=[a[0]];ri+=[a[1]]
    print(f'f1:{np.mean(k)}')
    print(f'right:{np.mean(ri)}')
'''2.0
f1:0.4727188141105084
right:0.1716669503622927'''
