import fc
import pandas as pd,numpy as np
file_list=['rb_min','c_min','ag_min']#'c_h.csv'#
def main(file_name):
    df = pd.read_csv('.\data\\'+file_name+'.csv')
    c=df['close_Rollover'].to_numpy()
    #cusum_0=df['CUSUM_000'].to_numpy()
    #cusum_1=df['CUSUM_888'].to_numpy()
    date=df['date'].to_numpy()
    c_diff=np.diff(c)
    date=date[1:]
    date=date*10000%10000
    crystal=[0]*1440
    crystal_abs=[0]*1440
    for i,k in enumerate(c_diff):
        crystal[round(date[i]//100*60+date[i]%100)]+=k
        crystal_abs[round(date[i]//100*60+date[i]%100)]+=abs(k)
    date_lon=len(c)//345
    return np.array([crystal,crystal_abs])/date_lon
crystal_list=[]
for i in file_list:
    crystal_list+=[main(i)]
crystal_list=np.array(crystal_list)
print(np.corrcoef(crystal_list[:,0,500:])[0][1])
print(np.corrcoef(crystal_list[:,1,500:])[0][1])
fc.plot(crystal_list[:,0],k=1)
fc.plot(crystal_list[:,1],k=1,zoom='auto')

    
