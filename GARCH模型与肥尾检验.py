import pandas as pd,numpy as np
import fc
from scipy.stats import kurtosis
file_list='.\data\c_day.csv'
df = pd.read_csv(file_list)
c=df['close_Rollover'].to_numpy()
#肥尾检验
k=kurtosis(np.diff(c))
if k > 3:
    print("数据分布具有厚尾特征")
else:
    print("数据分布符合正态分布或其他理论分布")
####
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

# 检验序列的平稳性
def check_stationarity(data):
    adf_result = adfuller(data)
    kpss_result = kpss(data)
    print(adf_result[1],kpss_result[1])
    if adf_result[1] < 0.05:# and kpss_result[1] > 0.05:
        return True
    else:
        return False

# 检验序列的自相关性
def check_autocorrelation(data):
    plot_acf(data)
    plot_pacf(data)
    
# 检验序列的异方差性
def check_heteroscedasticity(data):
    lbvalue, pvalue = acorr_ljungbox(np.array(fc.FDD_d(c,d=0.92 ))**2, lags=[10],return_df=False)
    if pvalue < 0.05:
        return True
    else:
        return False

# 拟合GARCH模型
def fit_garch_model(data):
    model = arch_model(data, p=1, q=1)
    result = model.fit()
    return result

# 检验序列是否适用于GARCH模型
def check_garch(data):
    if not check_stationarity(data):
        print("序列不平稳")
    else:
        check_autocorrelation(data)
        if not check_heteroscedasticity(data):
            print("序列方差稳定")
        else:
            result = fit_garch_model(data)
            print(result.summary())
            
# 使用示例
import numpy as np
import pandas as pd

# 生成随机序列
np.random.seed(1)

# 检验序列是否适用于GARCH模型
check_garch(np.array(fc.FDD_d(c,d=0.92 )))
from scipy.stats import normaltest

# 检验GARCH模型是否满足条件
def check_garch(data):
    if not check_stationarity(data):
        print("序列不平稳")
    else:
        check_autocorrelation(data)
        if not check_heteroscedasticity(data):
            print("序列方差稳定")
        else:
            model = arch_model(data, p=1, q=1)
            result = model.fit()
            # 检验参数是否满足非负性约束
            if (result.params >= 0).all():
                print("是：参数满足非负性约束")
            else:
                print("非：参数不满足非负性约束")
            # 检验模型拟合效果
            if result.pvalues[1] < 0.05:
                print("非：存在残差自相关性")
            else:
                print("是：不存在残差自相关性")
            if abs(result.resid.mean()) < 0.1:
                print("是：残差的均值接近于0")
            else:
                print("非：残差的均值不接近于0")
            if check_stationarity(result.resid**2):
                print("是：残差方差稳定")
            else:
                print("非：残差方差不稳定")
            # 检验残差序列是否呈现高斯分布
            if normaltest(result.resid)[1] > 0.05:
                print("是：残差序列呈现高斯分布")
            else:
                print("非：残差序列不呈现高斯分布")
            # 输出模型的拟合结果
            print(result.summary())
check_garch(np.array(fc.FDD_d(c,d=1 )))
