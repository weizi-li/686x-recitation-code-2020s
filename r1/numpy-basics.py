## Code for 6.86x recitation
## NOTE: the following script needs Numpy
## Practice: try to comment out each section to see the results

import numpy as np

### two ways of creating a matrix using Numpy
array1 = np.array([3,2,5])
array2 = np.array([1,-3,2])
array3 = np.array([5,-1,4])
m1 = np.array([array1,array2,array3])
m2 = np.array([[3,2,5],[1,-3,2],[5,-1,4]])
# print("m1:")
# print(m1)
# print("m2:")
# print(m2)


### creating certain matrix using Numpy build-in functions
m3 = np.random.rand(2,3)
# print("m3:")
# print(m3)
m4 = np.identity(4) # identity matrix
# print("m4:")
# print(m4)


### matrix arithmetics
print("m1:")
print(m1)
print("m1+1:")
print(m1+1)

print("m1:")
print(m1)
print("m1*2:")
print(m1*2)

m5 = np.array([[1,2],[3,4]])
m6 = np.array([[1,2],[3,4]])
#print(m5*m6) ## wrong!! verify this by hand!
print(np.matmul(m5,m6))


### access matrix elements and shape
print("m3:")
print(m3)
print(m3.shape)
print(m3.shape[0]) # access shape elements
print(m1[2][2]) # access matrix elements


### get the transpose of a matrix
print(m1.T)
print(m1.transpose())


### compute the determinant of a matrix
print(np.linalg.det(m1))


# ### compute the inverse of a matrix
m1_inv = np.linalg.inv(m1)
print(np.matmul(m1,m1_inv))


# ### solve a linear system of equations
A = np.array([[4,3],[-10,-2]])
B = np.array([-13,5])
A_inv = np.linalg.inv(A)
X = np.matmul(A_inv,B)
print(X)

