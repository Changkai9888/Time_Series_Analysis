import pandas as pd,numpy as np
import fc
file_list='.\data\c_min.csv'
df = pd.read_csv(file_list)
close=df['close'].to_numpy()
c_roll=df['close_Rollover'].to_numpy()
FFD_0=df['FFD_2d_sect_mean_8055'].to_numpy()
FFD_1=df['FFD_2d_sect_diff_8055'].to_numpy()
FFD_2=df['FFD_2d_sect_mean_7035'].to_numpy()
FFD_3=df['FFD_2d_sect_diff_7035'].to_numpy()
t15=df['time15'].to_numpy()
####
def first_index_above(arr, value,k=1):
    # 使用 torch.nonzero 找到所有大于或小于特定值的索引
    above_indices = np.argwhere(arr > value) if k==1 else np.argwhere(arr < value)
    # 如果没有大于特定值的索引，则返回 -1
    if len(above_indices) == 0:
        return len(arr)
    # 返回第一个大于特定值的索引
    return above_indices[0][0]
def med1():
    tag=[]
    for i in range(len(close)):
        c=c_roll[i:i+int(step)]
        imposs=set()
        if len(c)<int(step):
            tag+=[0];continue
        p_out=first_index_above(c, c[0]+alfa_profit,k=1)
        p_loss=first_index_above(c, c[0]-alfa_loss,k=-1)
        n_out=first_index_above(c, c[0]-alfa_profit,k=-1)
        n_loss=first_index_above(c, c[0]+alfa_loss,k=1)
        if p_out<p_loss:# or (p_loss==len(c) and n_loss!=len(c)):
            tag+=[1]
        elif n_out<n_loss:# or (n_loss==len(c) and p_loss!=len(c)):
            tag+=[-1]
        else: tag+=[0]
    return tag
def get_cxy_3(alfa_profit_=6,#止盈止损系数
                    alfa_loss_=6,#止盈止损系数
                    step_=30):#最大持仓系数)
    global alfa_profit,alfa_loss,step
    alfa_profit,alfa_loss,step=alfa_profit_,alfa_loss_,step_
    time_diff=np.insert(np.diff(t15),0,0)
    tag=np.array(med1()).astype(float)
    tag[tag==0]=0.01
    time_tag= [1,11,17]
    for i in range(len(tag)):
        if time_diff[i]==0 or not(t15[i] in time_tag):
            tag[i]=0
    tag_=tag
    c=np.array(c_roll.tolist()[:-int(step)])
    x=np.column_stack((t15,FFD_0, FFD_1,FFD_2, FFD_3,close%10))[:-int(step)]
    y=np.array(tag_[:-int(step)])
    for i in range(len(x)):
        if min(abs(x[i]))>0:
            break
    c,x,y=c[i:],x[i:],y[i:]
    #print(f"time_tag:{time_tag}");print(f"file_list:{file_list}");
    return c,x,y
