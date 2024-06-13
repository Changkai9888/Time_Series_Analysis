import numpy as np
import pywt
import matplotlib.pyplot as plt

# 生成信号
t = np.linspace(0, 1, 200, endpoint=False)
sig1 = np.sin(2 * np.pi * 7 * t) + np.cos(2 * np.pi * 15 * t)
sig2 = np.sin(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 10 * t)
sig = sig1 + sig2

# 进行小波变换
coeffs = pywt.wavedec(sig, 'db4', level=3)
cA3, cD3, cD2, cD1 = coeffs

# 绘制小波分解系数
plt.figure(figsize=(8, 6))
plt.subplot(411)
plt.plot(cA3)
plt.title('Approximation coefficients')
plt.subplot(412)
plt.plot(cD3)
plt.title('Detail coefficients, level 3')
plt.subplot(413)
plt.plot(cD2)
plt.title('Detail coefficients, level 2')
plt.subplot(414)
plt.plot(cD1)
plt.title('Detail coefficients, level 1')
plt.tight_layout()
plt.show()