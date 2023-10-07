import unittest
import numpy as np

shape = [2, 3]
#data = [1., 2., 3., 4., 5., 6.]
#f_data = np.arange(6, dtype=np.float32)+1
#int_data = np.arange(6, dtype=np.int32)+1
f_data = [[[1.,2.,3.]],[[4.,5.,6.]], [[0.4, 4.6, 2.1]]]
#int_data = [[1,2,3],[4,5,6]]
#f_data = [1.,2.,3.,4.,5.,6.]
int_data = [1,2,3,4,5,6]
offset=4

#tensor_float32 = unittest.TensorFloat32(shape, data, offset, devie='cuda')
#tensor_int = unittest.TensorInt(shape, data, offset, devie='cuda')
print(f'{f_data=}')
tensor_float32 = unittest.TensorFloat32(data=f_data, offset=offset, device='cuda')
#tensor_int32 = unittest.TensorInt(shape, int_data, offset)

f_shape = tensor_float32.shape()
f_size = tensor_float32.size()
f_strides = tensor_float32.strides()

#int_shape = tensor_float32.shape()
#int_size = tensor_float32.size()
#int_strides = tensor_float32.strides()

#print(f'{f_shape=}')
#print(f'{f_strides=}')
#print(f'{f_size=}')


#print(f'{int_shape=}')
#print(f'{int_strides=}')
#print(f'{int_size=}')


