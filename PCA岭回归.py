from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

# 加载数据集
data = fetch_california_housing().data
target = fetch_california_housing().target

# 进行 PCA 降维
pca = PCA(n_components=5)
data = pca.fit_transform(data)

# 进行岭回归
ridge = Ridge(alpha=0.1)
ridge.fit(data, target)

# 预测结果并计算 R 平方值
predictions = ridge.predict(data)
r2 = r2_score(target, predictions)

print("R^2 score: {:.2f}".format(r2))
######
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 加载数据集
data = fetch_california_housing(as_frame=True).frame

# 划分特征和目标变量
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

# 对特征进行PCA降维
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 构建线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 预测测试集结果并计算R平方值
y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)

print('R平方值:', r2)
