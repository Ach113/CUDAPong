import numpy as np

# array1 = np.array([6,2])
# array2 = np.array([2,5])

array1 = np.array([i for i in range(16)])
array2 = np.array([i+1 for i in range(16)])

print(array1)
print(array2)

print("Outer Product of the two array is:")
result = np.outer(array1, array2)
print(result)
