import 数据读取,尾数统计
import fc
import numpy as np,random
def corr_tail(step=1000,mod=10):
    c=数据读取.get_close(file_list=['c_min1_c']#['rb_min1_c']
                     )
    corr_list=[]
    for i in range(len(c)//step-1):
        corr_list+=[np.corrcoef(尾数统计.tail(c[i*step:(i+1)*step],mod)[0],
                               尾数统计.tail(c[(i+1)*step:(i+2)*step],mod)[0])[0, 1]]
    return corr_list
def research0102():
    list=[]
    for i in range(1,11):
        list+=[corr_tail(step=i*1000,mod=10)]
    print([round(np.mean(np.array(i)),2) for i in list])
    fc.plot([round(np.mean(np.array(i)),2) for i in list])
    lol=[]
    for k in range(1,11):
        list=[]
        for i in range(5,16):
            list+=[corr_tail(step=k*1000,mod=i)]
        lol+=[[round(np.mean(np.array(i)),2) for i in list]]
    fc.plot(lol,k=1)
    return
def research03():
    def evg(step=1000,mod=10):
        c=数据读取.get_close(file_list=['c_min1_c']#['rb_min1_c']
                         )
        lit=[];
        #建立尾数统计集合
        for i in range(len(c)//step):
            lit+=[尾数统计.tail(c[i*step:(i+1)*step],mod)[0]]
        rec=[]
        for i in range(1000):
            #按顺序计算平均值
            re=[]
            for i in range(len(lit)-1):
                re+=[np.corrcoef(lit[i],lit[i+1])[0, 1]]
            rec+=[round(np.mean(re),4)]
            #打乱lit1000次，计算平均值
            random.shuffle(lit)
        print(f'step={int(step/1000)}k,mod={mod}时，连续顺序的相关性：{rec[0]}；1000次随机顺序的相关性：{np.mean(rec[1:]):.4f}')
        return
    import itertools
    for i, k in itertools.product(range(1,11), range(5,16)):
        evg(step=1000*i,mod=k)
    return
