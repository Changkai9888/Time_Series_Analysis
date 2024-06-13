import fc
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
file_list='.\data\c_h.csv'#c_day_cusum_c.csv'#
df = pd.read_csv(file_list)
c=df['close_Rollover'].to_numpy()
cusum_0=df['CUSUM_000'].to_numpy()
cusum_1=df['CUSUM_888'].to_numpy()
date=df['date'].to_numpy()
date=date*100%100
####
def main():
    X=np.column_stack((cusum_0,cusum_1,cusum_0-cusum_1,date))[:-1]
    y=np.diff(c)
    X, X_test, y, y_test =train_test_split(X, y, test_size=0.2)
    #X, X_test, y, y_test= X[:-len(y)//4],X[-len(y)//4+3:],y[:-len(y)//4],y[-len(y)//4+3:]
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
    X_test_re,y_test_re,y_pred_re=X_test[abs(y_pred-np.mean(y_pred))>2*np.std(y_pred)],\
    y_test[abs(y_pred-np.mean(y_pred))>2*np.std(y_pred)],\
    y_pred[abs(y_pred-np.mean(y_pred))>2*np.std(y_pred)]
    # 输出模型评估指标
    #print("Training set score: {:.4f}".format(rf_reg.score(X_test_re,y_test_re)))
    #print(f'相关性系数为：{np.corrcoef(y_pred_re,y_test_re)[0, 1]}')
    #fc.plot([y_test,y_pred],k=1)
    """pos0=[]
    for i in y_pred:
        pos0+=[int((i-np.mean(y_pred))/np.std(y_pred))]
    '''pos=[pos0[0]]
    for i in pos0[1:]:
        pos+=[pos[-1]+int((i-pos[-1])/2)]'''
    '''pos=[]
    for k in pos0:
        if abs(k)>=2:
            pos+=[fc.sig(k)]
        else:
            pos+=[pos[-1] if len(pos)!=0 else 0]'''
    pos=pos0.copy()
    for i, k in enumerate(pos):
        if abs(k)>=2:
            pos[i]=fc.sig(k)
        else:
            pos[i]=0#单根线
    right_list=fc.get_right_diff(y_test,np.array(pos),cost=0)
    #fc.plot(right_list)
    #print(right_list[-1])"""
    return np.corrcoef(y_pred_re,y_test_re)[0, 1]
