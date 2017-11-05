import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
#print(data)
#根据是否
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
#print(positive)
#print(negative)
fig, ax = plt.subplots(figsize=(12,8))
# print(fig)
# print("------------------------------")
# print(ax)
#其中c代表颜色，marker代表的是形状，label标识的标签，scatter中的前两个参数代表是的在坐标系的中的x轴跟y轴
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
#设置x，y轴的标识
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

#plt.show()
#定义线性回归假设函数
def sigmoid(z):
    return 1/(1+np.exp(-z))
nums = np.arange(-10, 10, step=1)
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
#plt.show()
#定义代价函数
def cost(theta,X,y):
    #将传进来的theta,X,y转成矩阵的形式
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    return np.sum(first-second)/(len(X))
   #加上一列（这是规矩，为了常数那一列）
data.insert(0,"Ones",1)
#进行切片操作，提取出X跟y来
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
#进行将X跟y转换成数组
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(cols-1)#源文件直接是一个3
# print("---------------------")
# a = int(theta.ravel().shape[0])
# print(a)
# print("---------------------")
#print(data)
#计算代价函数的value
# a=cost(theta, X, y)
# print(a)
#定义梯度下降的函数
def gradient(theta,X,y):
    #转化成矩阵的形式
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    #设置
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    #在进行梯度下降的时候，进行对中间变量的迭代，因为不是最终的代价函数值，使用error
     #来表示
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error,X[:,i])
        grad[i] = np.sum(term)/len(X)
    return grad
# b=gradient(theta,X,y)
# print(b)
#使用SciPy's truncated newton(TNC)实现寻找最优参数,这个函数就是干这个的 不要问为什么
result = opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,y))
#该函数实现用所学习到参数theta来为数据集X输出预测，然后，我们可以使用这个函数来给我们的分类器的训练精度进行打分、
def predict(theta,X):
    probability = sigmoid(X*theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
theta_min = np.matrix( result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))