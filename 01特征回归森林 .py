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
####
df_X=df.iloc[:, 2:][:-1]
X_orig=df_X.values;
####
time_self_name=df_X['time_self_name'].to_numpy().reshape(-1,1)
Volat=df_X['Volat'].to_numpy().reshape(-1,1)
ATR=df_X['ATR'].to_numpy().reshape(-1,1)
Thold=np.concatenate((ATR,ATR**2,Volat,time_self_name**2), axis=1)
X0=X_orig
#X0=Thold#01模式
def main():
    train_idx, test_idx = train_test_split(np.arange(X0.shape[0]), test_size=0.2, shuffle=True)
    #train_idx, test_idx =np.arange(len(train_idx)),len(train_idx)+np.arange(len(test_idx))
    X_train = X0[train_idx]
    X_test = X0[test_idx]
    y_train = abs(y0[train_idx])
    y_test =  abs(y0[test_idx])
    y0_test = y0[test_idx]
    ####
    global rf_reg
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
    
    # 输出模型评估指标
    '''print(f'R^2:{r2_score(y_test,y_pred )}')
    print(f'corr：{np.corrcoef(y_pred,y_test)[0, 1]}')
    print('F1 score:', f1_score(y_test, y_pred, labels=[1,-1],average= 'weighted'))
    print('accuracy_score:',accuracy_score(y_test, y_pred))
    fc.plot([y_test,y_pred],k=1)
    fc.plot(fc.get_right_diff(y0_test,y_pred,cost=0))'''
    return r2_score(y_test,y_pred ),np.corrcoef(y_pred,y_test)[0, 1]
k=[];ri=[]
for i in range(10):
    a=main();print(i)
    k+=[a[0]];ri+=[a[1]]
print(f'R^2:{np.mean(k)}')
print(f'corr:{np.mean(ri)}')
fc.plot_tree(rf_reg)
