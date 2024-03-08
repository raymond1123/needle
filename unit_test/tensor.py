import numpy as np
import tensor as ts

# Define shape, strides, and offset
shape = [2, 3]
strides = [3, 1]
offset = 0

# Create a NumPy array with a specific data type (float32 in this case)
arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7., 8., 9.]], 
        dtype=np.float32)

# Create an instance of the FloatTensor class
ft1 = ts.Tensor(arr, ts.BackendType.CUDA)
ft2 = ts.Tensor(arr, ts.BackendType.CPU)

print(ft1.shape())
print(ft2.shape())

out1 = ft1.to_numpy()
out2 = ft2.to_numpy()

print(out1)
print(out2)

