import numpy as np
import pandas as pd
def get_close(file_list=['c_min1_c']#['rb_min1_c']
              ,get_data=0):
    "得到研究数据"
    data=[]
    for i in file_list:
        file=i+'.txt'
        f = open('./data/'+file,encoding = "ANSI")
        a=eval(f.readlines()[0])
        a_=a[-1];a=a[:-1];
        if (len(a)%a_!=0):
            print('数据出错！');return
    for i in range(a_):
        data+=[a[i::a_]]
    close=np.array([float(i) for i in data[0]])
    if get_data==0:
        return close
    else:
        return data
####
'''c=get_close()
import matplotlib.pyplot as plt
plt.scatter(range(len(c)), c, s=0.1, color='k')
plt.show()'''
def save_csv():
    data=get_close(file_list=['ag_min1_c'] ,get_data=1)
    date=np.array([float(i) for i in data[0]])
    close_Rollover=np.array([float(i) for i in data[1]])
    #CUSUM_000=np.array([float(i) for i in data[2]])
    #CUSUM_888=np.array([float(i) for i in data[3]])
    df = pd.DataFrame({'date': date,
                       'close_Rollover': close_Rollover,
                       })
    df.to_csv('./data/ag_min.csv', mode='a',index=False)
    return
def save_custom():
    p=['Datetime', 'C_Rollover', 'Volat', 'C_in_HL', 'C_normal', 'Spread_normal',\
       'CUSUM_0', 'CUSUM_8', 'Slope', 'LR_Std', 'LR_Value', 'LR_Value_normal', \
       'X_std', 'Skew', 'Kurt', 'Imba_IN_normal', 'Imba_OUT_normal', 'Coeff_HV', \
       'Coeff_LV', 'time_self_name', 'time_uper_name', 'Mod_c', 'Oneside_Last',\
       'Peace_Last', 'Boom_Last', 'Highest_past', 'Lowest_past','ATR']
    #持仓量相关
    data0=get_close(file_list=['0-26'] ,get_data=1)
    data=np.array(data0).astype(float)
    df = pd.DataFrame({p[i]:data[i] for i in range(len(p))})
    df.to_csv('./data//c_day_all.csv', mode='a',index=False)
    return
