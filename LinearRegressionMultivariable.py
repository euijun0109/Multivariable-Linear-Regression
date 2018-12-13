import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class LinearRegressionMultivariable:
    def __init__(self, alpha, n):
        self.thetas = np.zeros((n, 1), dtype= float)
        self.m = 0
        self.n = n
        self.cost = np.zeros((n, 1), dtype= float)
        self.alpha = alpha

    def hypothesis(self, x):
        return self.thetas.T.dot(x.T)

    def costs(self, x, y):
        self.cost = ((self.hypothesis(x).T - y).T.dot(x)).T

    def GD(self):
        self.thetas -= self.alpha * (1/self.m) * self.cost

    def calculate(self, x, y):
        self.costs(x, y)
        self.GD()
        print("costs:", end=" ")
        for cos in self.cost:
            print(cos[0], ",", end=" ")
        print(" ")
        self.cost = np.zeros((self.n, 1), dtype= float)
        print("thetas:", end=" ")
        for theta in self.thetas:
            print(theta[0], ",", end=" ")
        print('\n')

    def run(self, i, x, y):
        self.m = len(y)
        for _ in range(i):
            self.calculate(x, y)
        if self.n == 3:
            self.display3D(x, y)
        elif self.n == 2:
            self.display2D(x, y)
        else:
            pass

    def resFunction(self, x1, x2):
        return self.thetas[0][0] + self.thetas[1][0] * x1 + self.thetas[2][0] * x2

    def display3D(self, x1, y1):
        x = y = np.arange(-10.0, 10.0, 1)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self.resFunction(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        X1 = []
        Y1 = []
        Z1 = y1.tolist()
        for row in x1:
            X1.append(row[1])
            Y1.append(row[2])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X1, Y1, Z1, color='red')
        ax.plot_surface(X, Y, Z, alpha=0.5, color='grey')
        ax.legend()
        plt.show()

    def display2D(self, x, y):
        X1 = []
        Y1 = y.T.tolist()[0]
        for row in x:
            X1.append(row[1])
        X = [0, X1[self.m - 1] + 1]
        Y = [self.thetas[0], self.thetas[0] + self.thetas[1] * (X1[self.m - 1] + 1)]
        plt.plot(X1, Y1, "ro", label="data")
        plt.plot(X, Y, label="best fit line")
        plt.legend()
        plt.show()