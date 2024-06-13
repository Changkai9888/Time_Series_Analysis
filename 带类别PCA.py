import numpy as np
from sklearn.datasets import load_iris

# 加载Iris数据集
iris = load_iris()
X = iris.data  # 数据
y = iris.target  # 标签

# 计算数据集的均值
mean = np.mean(X, axis=0)

# 计算类内散布矩阵
Sw = np.zeros((X.shape[1], X.shape[1]))
for i in range(3):
    Xi = X[y == i]
    mean_i = np.mean(Xi, axis=0)
    Sw += np.dot((Xi - mean_i).T, (Xi - mean_i))

# 计算类间散布矩阵
Sb = np.dot((np.mean(X[y == 0], axis=0) - mean).reshape(-1, 1),
            (np.mean(X[y == 0], axis=0) - mean).reshape(1, -1))
Sb += np.dot((np.mean(X[y == 1], axis=0) - mean).reshape(-1, 1),
             (np.mean(X[y == 1], axis=0) - mean).reshape(1, -1))
Sb += np.dot((np.mean(X[y == 2], axis=0) - mean).reshape(-1, 1),
             (np.mean(X[y == 2], axis=0) - mean).reshape(1, -1))

# 计算特征值和特征向量
eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))

# 按照特征值排序，取前两个最大的特征向量
eig_pairs = [(eig_vals[i], eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs.sort(reverse=True)
w = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))

# 计算降维后的数据
X_lda = np.dot(X, w)

# 可视化结果
import matplotlib.pyplot as plt

plt.figure()
colors = ['red', 'green', 'blue']
labels = ['Setosa', 'Versicolor', 'Virginica']
for i in range(3):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], c=colors[i], label=labels[i])
plt.legend()
plt.show()
#########
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

# 加载iris数据集
iris = load_iris()
X = iris.data
y = iris.target

# 创建LDA模型，将样本从4维降至2维
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
for i in range(3):
    plt.scatter(-X_lda[y == i, 0], -X_lda[y == i, 1], c=colors[i], label=labels[i])
plt.legend()
plt.show()
