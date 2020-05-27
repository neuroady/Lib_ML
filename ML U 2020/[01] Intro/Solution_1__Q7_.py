import numpy as np

# [Q7.1] Compute the matriX product of two matrices
X = np.array([ [5, 8], [2, 1], [6, 7] ])
y = np.array([ [5, 8, 3], [2, 1, 9] ])


print("Q7.1\nX:\n", X, "\n")
print("\nY:\n", y , "\n\n")

print("Shapes:\n")
print(" X : ", X.shape )
print("\n Y : ", y.shape)
Z = np.dot(X, y)
print("\n\n X.Y = \n", Z)


# [Q7.2] Find the Eigen values of a matriX
del X
X = np.array( [ [1, 2], [2, 4] ])
eigVal_X, eigVec_X = np.linalg.eig(X)

print("\nQ7.2\nThe matrix is :\n", X)
print("\nEigen values of the matrix are:\n", eigVal_X)
print("\nEigen vector of the matrix is :\n", eigVec_X)
