import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv("ex2data2.txt", names=['Test 1', 'Test 2', 'Accepted'])

positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()

# 看图拟合曲线 形如：1 + x1 + x1^2  + x1^2 * x2 + ....... + x1^4 * x2 ^3 = 0 (加上参数就是在拟合决策边界决策边界)

degree = 5
x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)


# 逻辑回归需要的sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 逻辑回归 正则化代价函数
def costReg(theta, x, y, learningRate):
    first = -y.T @ np.log(sigmoid(x @ theta.T))
    second = (1 - y).T @ np.log(1 - sigmoid(x @ theta.T))
    reg = (learningRate / (2 * len(x))) * np.sum(np.power(theta[:], 2))
    return (first - second) / len(x) + reg


# 正则化梯度下降函数
def gradientReg(theta, x, y, learningRate):
    parameters = theta.shape[0]
    grad = np.zeros(parameters)
    error = sigmoid(x * theta.T) - y  # @ 与 * 区分

    for i in range(parameters):
        term = sigmoid(x @ theta.T)
        term = term.reshape(term.shape[0], 1) - y
        term = term.T @ x[:, i].reshape(term.shape[0], 1)

        if i == 0:
            grad[i] = term / len(x)
        else:
            grad[i] = term / len(x) + ((learningRate / len(x)) * theta[i])

    return grad


# 预测最后的准确率
def predict(theta, x):
    probability = sigmoid(x @ theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# 初始化数据
cols = data.shape[1]
x = data.iloc[:, 1:cols]
y = data.iloc[:, 0:1]
x = np.asarray(x.values)
y = np.asarray(y.values)
theta = np.zeros(cols - 1)
learningRate = 1
# fprime 梯度函数，注意是梯度函数即可，不需要给出迭代的算子；func 代价函数；x0：初始化参数（最优解参数）；args：fprime需要的参数，除了x0其他都要写进去
res = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(x, y, learningRate))
theta_min = res[0]  # 优化参数
predictions = predict(theta_min, x)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
