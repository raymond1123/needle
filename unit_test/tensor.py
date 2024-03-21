import numpy as np
import tensor as ts

# Define shape, strides, and offset
shape = [2, 3]
strides = [3, 1]
offset = 0

# Create a NumPy array with a specific data type (float32 in this case)
arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7., 8., 9.]], 
        dtype=np.float32)

# Create a NumPy array with a specific data type (float32 in this case)
brr = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1., 1., 1.]], 
        dtype=np.float32)

# Create an instance of the FloatTensor class
#print('ft1:')
ft1 = ts.Tensor(np_array=arr, backend=ts.BackendType.CUDA)
#print(ft1.shape())
out1 = ft1.to_numpy()
#print(out1)
#print('\n')

#print('ft2:')
#ft2 = ts.Tensor(arr, ts.BackendType.CPU)
#print(ft2.shape())
#out2 = ft2.to_numpy()
#print(out2)

ft3 = ts.Tensor(brr, ts.BackendType.CUDA)
ft4 = ft1+ft3
#print(ft4.to_numpy())


ft5 = ft4+ft3
print(ft5.to_numpy())

