import fc
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import export_text
from joblib import parallel_backend
import time
import matplotlib.pyplot as plt
import graphviz,os,numpy as np
import importlib,torch
import 样本标签获得
#from sklearn.datasets import load_iris
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz/bin/'
def pot_tree(tree,x=[],y=[]):
    # 输出预测结果和准确率
    if x!=[] and y!=[]:
        y_pred = tree.predict(x)
        print("Predictions: ", y_pred)
        print("Accuracy: ", tree.score(x, y))
    # 打印决策树的结构和具体参数
    print(tree.tree_.max_depth)
    print(tree.tree_.n_features)
    print(tree.tree_.n_classes)
    print(tree.tree_.threshold)
    print(tree.tree_.value)
    print(tree.tree_.feature)
    print(tree.tree_.children_left)
    print(tree.tree_.children_right)
    # 使用plot_tree函数可视化决策树
    fig, ax = plt.subplots(figsize=(18, 18))
    plot_tree(tree, ax=ax, feature_names=np.array(range(1,14)), filled=True
              ,class_names=['-1,','1'])
    plt.show()
####
def trees_try(n_split=5,get_data_func=样本标签获得.get_cxy_3(alfa_profit_=25,#止盈止损系数
                                alfa_loss_=20,#止盈止损系数
                                step_=50)):
    rfc = RandomForestClassifier(max_features=1,
                                 n_estimators=1000,
                                 min_weight_fraction_leaf=0.05,
                                 class_weight='balanced_subsample',
                                 #class_weight={0: 0.1, 1: 1-p_n, -1: p_n},
                                 criterion='entropy',
                                 random_state=42)
    # 加载数据集
    #iris = load_iris();x, y = iris.data, iris.target
    #c,x,y=fc.load_temp('随机森林数据')
    c,x_,y_=get_data_func
    ###
    # 将数据集分为训练集和测试集
    p_n=(np.count_nonzero(y_==1)+np.count_nonzero(y_==-1))/len(y_)
    if p_n>0.005:
        x=x_[y_!=0];y=y_[y_!=0];y[abs(y)==0.01]=0
        #x_train, x_test, y_train, y_test = x[:-int(len(x)/5+10)],x[-int(len(x)/5):],y[:-int(len(x)/5+10)],y[-int(len(x)/5):]
        #x_train=x_train[y_train!=0];y_train=y_train[y_train!=0];
        #x_test=x_test[y_test!=0];y_test=y_test[y_test!=0];
        # 构建随机森林模型
        #p_n=np.count_nonzero(y_train==1)/(np.count_nonzero(y_train==1)+np.count_nonzero(y_train==-1))
        #scores = cross_val_score(rfc, x, y, cv=5)
        #print("随机森林交叉验证得分: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # 训练模型
        n_split=n_split
        cv=KFold(n_splits=n_split, shuffle=False)
        scores = cross_val_score(rfc, x, y, cv=cv)
        print(scores);print(np.mean(scores))
        rfc.fit(x[:-int(len(y)/n_split)-5], y[:-int(len(y)/n_split)-5])
        rfc.score(x[-int(len(y)/n_split):], y[-int(len(y)/n_split):])
        # 在测试集上进行预测
        #y_pred = rfc.predict(x_test)
        # 输出预测结果和准确率
        print("Accuracy_train: ", rfc.score(x[:-int(len(y)/n_split)-5], y[:-int(len(y)/n_split)-5]))
        print("Accuracy_test: ", rfc.score(x[-int(len(y)/n_split):], y[-int(len(y)/n_split):]))
        # 输出随机森林中每棵决策树的结构
        trees=rfc.estimators_
        tree=trees[0]
        #pot_tree(tree,x=x,y=y)
        y_pred = rfc.predict(x[-int(len(y)/n_split):])
        fc.plot([y[-int(len(y)/n_split):]+0.02,y_pred-0.02],k=1,zoom='auto')
    else:
        print(p_n)
#####
def grid_search():
    rec=0
    for i in range(1,6):
        for k in range(1,min(6,i+1)):
            for t in range(1,5):
                c,x,y=样本标签获得.get_cxy(alfa_profit_=5*i,#止盈止损系数
                                alfa_loss_=5*k,#止盈止损系数
                                step_=25*t)
                p_n=(np.count_nonzero(y==1)+np.count_nonzero(y==-1))/len(y)
                if p_n<0.005:
                    print(t);print(p_n);print(np.count_nonzero(y==1));print(np.count_nonzero(y==-1))
                    continue
                x=x[y!=0];y=y[y!=0]
                scores = cross_val_score(rfc, x, y, cv=5)
                print(str(5*i)+'_'+str(5*k)+'_'+str(25*t)+'_'+"随机森林交叉验证得分: %0.2f (+/- %0.2f)"% (scores.mean(), scores.std() * 2) +'_'+str(len(y)))
                print(np.count_nonzero(y==1));print(np.count_nonzero(y==-1))
                if scores.mean()>rec:
                    rec=scores.mean();result=[[rec,25*i,25*k,25*t]];
                if scores.mean()==rec:
                    result+=[[rec,25*i,25*k,25*t]]
    return
####
def Backtesting(n_split=5):
    alfa_profit_=25;alfa_loss_=20;step_=50
    c,x_,y_=样本标签获得.get_cxy(alfa_profit_=25,#止盈止损系数
                                alfa_loss_=20,#止盈止损系数
                                step_=50)
    x=x_[y_!=0];y=y_[y_!=0];y[abs(y)==0.01]=0
    #y_[abs(y_)==0.01]=0
    rfc = RandomForestClassifier(#max_features=1,
                                 n_estimators=100,
                                 min_weight_fraction_leaf=0.05,
                                 class_weight='balanced_subsample',
                                 #class_weight={0: 0.1, 1: 1-p_n, -1: p_n},
                                 criterion='entropy',
                                 random_state=42)
    n_split=n_split
    rfc.fit(x[:-int(len(y)/n_split)-5], y[:-int(len(y)/n_split)-5])
    y_pred=rfc.predict(x_)
    y_prob = rfc.predict_proba(x_)#标签概率
    pos=0;pos_lst=[0]*len(x_)
    right_list=[0]*len(x_)
    tim_rec=0
    for i in range(len(x_)):
        if i-150>0:
            right_list[i]+=pos*(c[i]-c[i-1]);pos_lst[i]=pos
            if pos==0 and y_pred[i]*np.mean(y_pred[i-100:i])>0.8 and i-tim_rec>100:
                pos=y_pred[i];c_rec=c[i];tim_rec=i;right_list[i]+=-2
            if pos!=0:
                if (c[i]-c_rec)*pos>25 or (c[i]-c_rec)*pos<-20 or i-tim_rec>50:
                    if y_pred[i]*np.mean(y_pred[i-100:i])>0.8 and pos==y_pred[i]:
                        c_rec=c[i];tim_rec=i;
                    else:
                        pos=0;right_list[i]+=-2;tim_rec=i;
    for i in range(len(x_)):
        if np.count_nonzero(y_[:i]!=0)>=len(y[:-int(len(y)/n_split)-5]):
            break
    right=fc.cal(right_list)
    print(right[-1])
    fc.plot([right[i:],c[i:],pos_lst[i:]],k=1,zoom='auto')
    #return right,pos_lst
    #right,pos_lst=right_test()
    return y_pred
def my_criterion(split_value, feature_idx, X, y):
    """
    自定义评价函数，最大化新的子节点中目标类别1在该子节点中的比率
    """
    # 将训练数据按照分裂阈值分为左右两个子集
    left_mask = X[:, feature_idx] < split_value
    right_mask = X[:, feature_idx] >= split_value
    left_y = y[left_mask]
    right_y = y[right_mask]
    # 计算新的左右子节点中目标类别1在该子节点中的比率
    left_ratio = np.mean(left_y == 1)
    right_ratio = np.mean(right_y == 1)
    # 计算节点分裂的效果
    improvement = left_ratio+right_ratio
    return improvement
#####孤立森林模型：异常值分析
def iso_trees():
    from sklearn.ensemble import IsolationForest
    isol= IsolationForest(random_state=0)
    n_split=1.1
    isol.fit(x_[:-int(len(y_)/n_split)-5], y_[:-int(len(y_)/n_split)-5])
    isol_scores = isol.decision_function(x_)
    isol.fit(x_,y_);
    isol_scores_real = isol.decision_function(x_)
    fc.plot([isol_scores_real,isol_scores],k=1,zoom='auto')
    return
###
#def Backtesting_02(n_split=3,alfa_profit=8,alfa_loss=4,step=30):
with parallel_backend('threading', n_jobs=14):
#if 1==1:
    # 多线程训练模型
    temp=time.time();
    alfa_profit=8;alfa_loss=4;step=30
    n_split=3;
    c,x_,y_=样本标签获得.get_cxy_3(alfa_profit_=alfa_profit,#止盈止损系数
                                alfa_loss_=alfa_loss,#止盈止损系数
                                step_=step)
    x=x_[y_!=0];y=y_[y_!=0];y[abs(y)==0.01]=0
    #y_[abs(y_)==0.01]=0
    rfc = RandomForestClassifier(max_features=1,
                                #max_depth=5,
                                n_estimators=10000,
                                min_weight_fraction_leaf=0.05,
                                class_weight='balanced_subsample',
                                #class_weight={0: 0.1, 1: 1-p_n, -1: p_n},
                                criterion='entropy',
                                random_state=42)
    #print(f"n_estimators:{rfc.n_estimators}");
    rfc.fit(x[:-int(len(y)/n_split)-5], y[:-int(len(y)/n_split)-5])

    y_pred=rfc.predict(x_)
    #y_prob=rfc.predict_proba(x_)
    '''y_pred=y_prob[:,2]-y_prob[:,0]
    y_pred[y_prob[:,1]>=y_prob[:,0]]=0
    y_pred[y_prob[:,1]>=y_prob[:,2]]=0
    y_pred[abs(y_pred)<0.1]=0'''
    '''y_pred[:]=0
    y_pred[y_prob[:,2]>0.5]=1
    y_pred[y_prob[:,0]>0.5]=-1'''
    pos=0;pos_lst=[0]*len(x_)
    tim_rec=0
    p_rec=0
    for i in range(len(x_)):
        if y_[i]!=0:
            pos=y_pred[i];tim_rec=i;p_rec=c[i]
        if i-tim_rec>step or (c[i]-p_rec)*pos>=alfa_profit or (c[i]-p_rec)*pos<=-alfa_loss:
            if pos*y_pred[i]<=0:
                pos=0
            elif pos*y_pred[i]>0:
                p_rec=c[i]; tim_rec=i 
        pos_lst[i]=pos
    ri_report=fc.get_right(c,np.array(pos_lst),cost=0.156)
    for i in range(len(x_)):
        if np.count_nonzero(y_[:i]!=0)>=len(y[:-int(len(y)/n_split)-5]):
            break
    print(f'{time.time()-temp}秒')
    print(f'参数n_split,alfa_profit,alfa_loss,step：{n_split,alfa_profit,alfa_loss,step}')
    print(ri_report[1][-1])
    print(ri_report[1][-1]-ri_report[1][i])
    pos_lst_diff=np.diff(np.array(pos_lst))[i:]
    print(f'开平共：{len(pos_lst_diff[pos_lst_diff!=0])}次')
    fc.plot([ri_report[1],[np.mean(ri_report[1])]*i],k=1)
#Backtesting_02(n_split=3,alfa_profit=8,alfa_loss=4,step=30)
