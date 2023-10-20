import unittest
import numpy as np

#data = [1., 2., 3., 4., 5., 6.]
a_data = [[[1.,2.,3.]],[[4.,5.,6.]], [[0.4, 4.6, 2.1]]]
b_data = [[[1.1,2.1,3.1]],[[4.1,5.1,6.1]], [[1.4, 1.6, 1.1]]]
#int_data = [[1,2,3],[4,5,6]]
#f_data = [1.,2.,3.,4.,5.,6.]
offset=4

#a = unittest.Tensor(data=a_data, offset=offset, device='cuda')
#b = unittest.Tensor(data=b_data, offset=offset, device='cuda')
#a = unittest.Tensor(data=a_data, device='cuda')
#b = unittest.Tensor(data=b_data, device='cuda')
#c = a+b
#d = a+b+c

#print(f'{a=}')
#print(f'{b=}')
#print(f'{c=}')
#print(f'{d=}')
#print(f'{a=}')
#print(f'{b=}')
#print(f'{c=}')

#numpy_a = a.numpy()
#numpy_b = b.numpy()
#numpy_c = c.numpy()
#numpy_d = d.numpy()

#print(f'{numpy_a=}')
#print(f'{numpy_b=}')
#print(f'{numpy_c=}')
#print(f'{numpy_d=}')

shape = (2, 3)
#ones = unittest.ones(shape).numpy()
ones = unittest.ones(shape)
print(f'{ones=}')

#f_shape = tensor_float32.shape()
#f_size = tensor_float32.size()
#f_strides = tensor_float32.strides()

