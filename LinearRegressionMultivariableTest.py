import LinearRegressionMultivariable as lrm
import numpy as np

x3 = np.array([[1, 3, 5], [1, 4, 3], [1, 6, 3], [1, 7, 2]], dtype=float)
y3 = np.array([[3], [4], [5], [6]], dtype=float)

x2 = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 8]], dtype=float)
y2 = np.array([[2], [4], [7], [6], [9], [11]], dtype=float)

LRM3D = lrm.LinearRegressionMultivariable(0.05355, 3)
LRM3D.run(70000, x3, y3)

print(LRM3D.predict(np.array([1, 3, 5])))

LRM2D = lrm.LinearRegressionMultivariable(0.01, 2)
LRM2D.run(10000, x2, y2)

#Capable of higher dimensions
