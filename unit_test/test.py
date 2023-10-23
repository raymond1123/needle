import Unittest as ndl
import numpy as np
import Unittest

def test_ewise_div():
    shape = (3, 2, 4)  # Change this to your desired shape
    np_a = np.random.uniform(0.0, 1.0, size=shape)
    np_b = np.random.uniform(0.0, 1.0, size=shape)

    ndl_a = ndl.Tensor(data=np_a.tolist(), device='cuda')
    ndl_b = ndl.Tensor(data=np_b.tolist(), device='cuda')
    ndl_c = ndl_a / ndl_b
    ndl_d = ndl_a / ndl_b / ndl_c

    np_c = np_a / np_b
    np_d = np_a / np_b / np_c

    np.testing.assert_allclose(ndl_c.numpy(), np_c,
                               rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(ndl_d.numpy(), np_d,
                               rtol=1e-5, atol=1e-5)

def test_ewise_mul():
    shape = (3, 2, 4)  # Change this to your desired shape
    np_a = np.random.uniform(0.0, 1.0, size=shape)
    np_b = np.random.uniform(0.0, 1.0, size=shape)

    ndl_a = ndl.Tensor(data=np_a.tolist(), device='cuda')
    ndl_b = ndl.Tensor(data=np_b.tolist(), device='cuda')
    ndl_c = ndl_a * ndl_b
    ndl_d = ndl_a * ndl_b * ndl_c

    np_c = np_a * np_b
    np_d = np_a * np_b * np_c

    np.testing.assert_allclose(ndl_c.numpy(), np_c,
                               rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(ndl_d.numpy(), np_d,
                               rtol=1e-5, atol=1e-5)

def test_ewise_minus():
    shape = (3, 2, 4)  # Change this to your desired shape
    np_a = np.random.uniform(0.0, 1.0, size=shape)
    np_b = np.random.uniform(0.0, 1.0, size=shape)

    ndl_a = ndl.Tensor(data=np_a.tolist(), device='cuda')
    ndl_b = ndl.Tensor(data=np_b.tolist(), device='cuda')
    ndl_c = ndl_a - ndl_b
    ndl_d = ndl_a - ndl_b - ndl_c

    np_c = np_a - np_b
    np_d = np_a - np_b - np_c

    np.testing.assert_allclose(ndl_c.numpy(), np_c,
                               rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(ndl_d.numpy(), np_d,
                               rtol=1e-5, atol=1e-5)

def test_ewise_add():
    shape = (3, 2, 4)  # Change this to your desired shape
    np_a = np.random.uniform(0.0, 1.0, size=shape)
    np_b = np.random.uniform(0.0, 1.0, size=shape)

    ndl_a = ndl.Tensor(data=np_a.tolist(), device='cuda')
    ndl_b = ndl.Tensor(data=np_b.tolist(), device='cuda')
    ndl_c = ndl_a + ndl_b
    ndl_d = ndl_a + ndl_b+ndl_c

    np_c = np_a + np_b
    np_d = np_a + np_b + np_c

    np.testing.assert_allclose(ndl_c.numpy(), np_c,
                               rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(ndl_d.numpy(), np_d,
                               rtol=1e-5, atol=1e-5)

def test_ones():
    shape = (2, 3)
    ndl_ones = ndl.ones(shape)
    np_ones = np.ones(shape)

    np.testing.assert_allclose(ndl_ones.numpy(), np_ones)

#test_ewise_add()
#test_ones()
#test_ewise_minus()

def test_topo_sort():
    # Test case 1
    a1, b1 = ndl.Tensor(np.asarray([[0.88282157]])), ndl.Tensor(np.asarray([[0.90170084]]))
    c1 = 3*a1*a1 + 4*b1*a1 - a1

    soln = np.array([np.array([[0.88282157]]),
                     np.array([[2.64846471]]),
                     np.array([[2.33812177]]),
                     np.array([[0.90170084]]),
                     np.array([[3.60680336]]),
                     np.array([[3.1841638]]),
                     np.array([[5.52228558]]),
                     np.array([[-0.88282157]]),
                     np.array([[4.63946401]])])

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c1])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

    # Test case 2
    '''
    a1, b1 = ndl.Tensor(np.asarray([[0.20914675], [0.65264178]])), ndl.Tensor(np.asarray([[0.65394286, 0.08218317]]))
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    soln = [np.array([[0.65394286, 0.08218317]]),
            np.array([[0.20914675], [0.65264178]]),
            np.array([[0.19040619]]),
            np.array([[1.53101102, 0.19240724]]),
            np.array([[0.44577898]]), np.array([[0.63618518]]),
            np.array([[1.90855553]]), np.array([[3.40855553]])]

    topo_order = [x.numpy() for x in ndl.autograd.find_topo_sort([c1])]

    assert len(soln) == len(topo_order)
    # step through list as entries differ in length
    for t, s in zip(topo_order, soln):
        np.testing.assert_allclose(t, s, rtol=1e-06, atol=1e-06)

    # Test case 3
    a = ndl.Tensor(np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]))
    b = ndl.Tensor(np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]))
    e = (a@b + b - a)@a

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([e])])

    soln = np.array([np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
                     np.array([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]),
                     np.array([[1.6252339, -1.38248184], [1.25355725, -0.03148146]]),
                     np.array([[2.97095081, -2.33832617], [0.25927152, -0.07165645]]),
                     np.array([[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]),
                     np.array([[1.53744921, -2.64392589], [0.17796981, 1.07906726]]),
                     np.array([[1.98898021, 3.51227226], [0.34285002, -1.18732075]])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)
    '''


