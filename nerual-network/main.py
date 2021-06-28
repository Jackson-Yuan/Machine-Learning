import numpy as np

from scipy.io import loadmat
import scipy.optimize as opt

data = loadmat("ex3data1.mat")


# 逻辑回归需要的sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 逻辑回归 正则化代价函数
def costReg(theta, x, y, learningRate):
    first = -y.T @ np.log(sigmoid(x @ theta.T))
    second = (1 - y).T @ np.log(1 - sigmoid(x @ theta.T))
    reg = (learningRate / (2 * len(x))) * np.sum(np.power(theta[:], 2))
    return (first - second) / len(x) + reg


# 梯度下降，全向量化表示
def gradient(theta, x, y, learningRate):
    term = sigmoid(x @ theta.T)
    term = term.reshape(term.shape[0], 1) - y
    grad = (term.T @ x) / len(x) + (learningRate / len(x)) * theta
    grad[0, 0] = (term.T @ x[:, 0]) / len(x)

    return grad.reshape(401)


# 多分类器
def one_vs_all(x, y, num_labels, learning_rate):
    rows = x.shape[0]
    params = x.shape[1]

    # 10个类别，所以需要10个分类器，即每一行存储每个分类器的训练参数，因样本有400个特征，额外再加一个常数项
    all_theta = np.zeros((num_labels, params + 1))

    # 补1
    x = np.insert(x, 0, values=np.ones(rows), axis=1)

    # 为每个分类器训练
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = opt.minimize(fun=costReg, x0=theta, jac=gradient, args=(x, y_i, learning_rate), method="TNC")
        all_theta[i - 1, :] = fmin.x

    return all_theta


# 根据多分类器的预测值，选取最大的一个
def predict_all(x, all_theta):
    rows = x.shape[0]
    params = x.shape[1]

    x = np.insert(x, 0, values=np.ones(rows), axis=1)

    h = sigmoid(x @ all_theta.T)

    # 每一行挑取最大的概率的索引，即为所要预测的数
    h_max = np.argmax(h, axis=1)

    return h_max


all_theta = one_vs_all(data["X"], data["y"], 10, 1)
max_predict = predict_all(data['X'], all_theta)
print(max_predict)