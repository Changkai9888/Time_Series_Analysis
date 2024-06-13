import pandas as pd,numpy as np
import fc
file_list='.\data\c_day.csv'
data = pd.read_csv(file_list)
c=data['close_Rollover'].to_numpy()
c_diff=[0]
for i in range(1,len(c)):
    c_diff+=[c[i]/c[i-1]-1]
c_diff=np.array(c_diff)
#等加权日收益率 绝对值 序列
###########
from pmdarima.arima import auto_arima
import pandas as pd
# 读取数据
sales_data =c_diff
# 使用自动ARIMA模型选择最佳ARIMA参数
#model = auto_arima(sales_data, seasonal=False, suppress_warnings=True);print(model.order)

import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import acf, pacf

# 可视化数据
'''plt.plot(sales_data)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()'''
######

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# 绘制ACF和PACF图
#acf_data = acf(sales_data, nlags=20, fft=False)
#pacf_data = pacf(sales_data, nlags=20, method='ols')
'''plot_acf(sales_data)
plot_pacf(sales_data)
plt.show()'''
######

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
def a(order=(11,0,19)):
    # 拟合ARIMA模型
    model = ARIMA(sales_data, order=order)
    results = model.fit()

    # 模型预测
    preds =results.predict(start=0, end=4191)
    #results.predict(start='2022-01-01', end='2022-12-31', dynamic=True)
    print(r2_score(sales_data,preds[:len(sales_data)]))
    return r2_score(sales_data,preds[:len(sales_data)])
# 可视化预测结果
if __name__=='__main__':
    plt.plot(preds, label='Predicted')
    plt.plot(sales_data, label='Actual')
    plt.legend()
    plt.show()

