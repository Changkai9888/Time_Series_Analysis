path='C:/Quant工坊'
import sys
sys.path.append(path)
import random as rd,numpy as np
import fc
def tail(close,mod=10):
    #bar=fc.make_easy(bar)
    'p10:尾数余数；p10_to_3：某个尾数的趋势概率'
    p10=[0]*mod
    for i in close:
        p10[round(i)%mod]+=1
    p10_to_3=[[0,0,0]]*mod
    p10_to_3=np.array(p10_to_3)
    for i in range(len(close)-1):
        this_number=round(close[i])%mod
        add_number=fc.sig(close[i+1]-close[i])
        if abs(add_number)<=1:
            p10_to_3[this_number][add_number+1]+=1
    p10_to_3=p10_to_3.astype('float')
    for i,k in enumerate(p10_to_3):
        p10_to_3[int(i)]=[k[0]/sum(k),k[1]/sum(k),k[2]/sum(k)]
    return p10,p10_to_3
#####
def compare(close,mod=10):
    num=len(close)
    p10_to_3=tail(close,mod)[1]
    "根据p10_to_3，模拟变化趋势。"
    a=[2500]
    for i in range(num):
        p=p10_to_3[round(a[-1]%mod)]
        a+=[a[-1]+rd.choices([-1,0,1],weights=p)[0]]
    fc.plot(a)
    b=[0]*mod
    for i in a:
        b[round(i%mod)]+=1
    fc.plot(b)

