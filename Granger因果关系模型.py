import numpy as np

# 生成数据
A = np.random.rand(100)
B = np.zeros_like(A)
B[40:] = A[:-40]

# 将A,B合并成一个2D array
data = np.vstack([A, B])

# 对数据进行SVD分解
U, s, V = np.linalg.svd(data, full_matrices=False)

# 选择前2个奇异值对应的左奇异向量构成新的特征向量
features = U[:, :2]

# 将数据投影到新的特征空间
proj = np.dot(features.T, data)

# 从投影后的数据中分离出A'和B'
A_prime, B_prime = proj
#B_prime = np.dot(U[:, 1:], np.dot(np.diag(s[1:]), V[1:, :]))
# 可视化结果
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

ax[0, 0].plot(A, color='b')
ax[0, 0].set_title('A')

ax[1, 0].plot(B, color='g')
ax[1, 0].set_title('B')

ax[0, 1].scatter(range(len(A_prime)), A_prime, color='b')
ax[0, 1].set_title("A'")

ax[1, 1].scatter(range(len(B_prime)), B_prime, color='g')
ax[1, 1].set_title("B'")

plt.tight_layout()
plt.show()
