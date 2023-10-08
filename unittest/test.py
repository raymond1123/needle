import unittest
import numpy as np

shape = [2, 3]
#data = [1., 2., 3., 4., 5., 6.]
#int_data = np.arange(6, dtype=np.int32)+1
a_data = [[[1.,2.,3.]],[[4.,5.,6.]], [[0.4, 4.6, 2.1]]]
b_data = [[[1.1,2.1,3.1]],[[4.1,5.1,6.1]], [[1.4, 1.6, 1.1]]]
#int_data = [[1,2,3],[4,5,6]]
#f_data = [1.,2.,3.,4.,5.,6.]
int_data = [1,2,3,4,5,6]
offset=4

#tensor_float32 = unittest.TensorFloat32(shape, data, offset, devie='cuda')
#tensor_int = unittest.TensorInt(shape, data, offset, devie='cuda')
a = unittest.TensorFloat32(data=a_data, offset=offset, device='cuda')
b = unittest.TensorFloat32(data=b_data, offset=offset, device='cuda')
c = a+b

print(f'{a=}')
print(f'{b=}')
print(f'{c=}')
#tensor_int32 = unittest.TensorInt(shape, int_data, offset)

#f_shape = tensor_float32.shape()
#f_size = tensor_float32.size()
#f_strides = tensor_float32.strides()

