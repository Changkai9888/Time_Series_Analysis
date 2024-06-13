path='C:/Quant数据库/数据正交化/'
import fc
from PIL import Image
import matplotlib.pyplot as plt
import random,numpy as np
#本程序结论：周期为5分钟，开仓的早上交易最为聚集。
def img(close):
    "10分钟单位，画出动价图"
    p=np.diff(close)
    ####
    c=[];k=0
    step=1200
    most=len(p)//step-1
    for i in range(len(p)):
        b=p[i]
        if k>=most and len(c[-1])==step:
            break
        if i%step==0:
            c+=[[]];k+=1
        if b>0:
            c[-1]+=[[255,0,0]]
        elif b==0:
            c[-1]+=[[100,100,100]]
        else:
            c[-1]+=[[0,0,255]]
    img=np.array(c)
    plt.imshow(img)
    plt.show()
def img2(bar):
    "以天为单位"
    c=[[]]
    for i in range(len(bar)):
        if i==0:
            c=[[[0,0,0]]]
        else:
            if bar[i][-2]<bar[i-1][-2]:
                while len(c[-1])<41630:
                    c[-1]+=[[0,0,0]]
                c+=[[[0,0,0]]]
            else:
                if bar[i][1]==bar[i-1][1]:
                    c[-1]+=[[100,100,100]]
                if bar[i][1]>bar[i-1][1]:
                    c[-1]+=[[255,0,0]]
                if bar[i][1]<bar[i-1][1]:
                    c[-1]+=[[0,0,255]]
    while len(c[-1])<41630:
        c[-1]+=[[0,0,0]]
    cs=[]
    for i in c:
        cs+=[i]*10
    img=np.array(cs)
    plt.imshow(img)
    plt.show()
