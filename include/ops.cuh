#ifndef __OPS_CUH__
#define __OPS_CUH__

#include "common.hpp"
#include "tensor.hpp"

namespace needle {
template <typename Dtype> class Value;
template <typename Dtype> class Tensor;
template <typename Dtype> class NDArray;

/*
template<typename Dtype>
class OP {
public:
    virtual Tensor<Dtype> compute(const std::vector<std::shared_ptr<Tensor<Dtype>>> &inputs)=0;
};
*/

template<typename Dtype>
class TensorOP {
public:
    TensorOP() = default;
    TensorOP(std::string op_name): _op_name(op_name) {}

    virtual NDArray<Dtype> compute(const std::vector<const Tensor<Dtype>*> &inputs)=0;

    //virtual Tensor<Dtype> call(const std::vector<Tensor<Dtype>&> &inputs) {
    //    return Tensor<Dtype>::make_from_op(std::shared_ptr<OP<Dtype>>(this), 
    //                                       inputs);
    //}

public:
    std::string _op_name;
};

template<typename Dtype>
void vec_add_wrapper(Dtype *a, Dtype *b, Dtype *c, size_t n);

template<typename Dtype>
class EwiseAdd: public TensorOP<Dtype> {
public:
    EwiseAdd() = default;
    EwiseAdd(std::string op_name): TensorOP<Dtype>(op_name) {}

    virtual NDArray<Dtype> compute(const std::vector<const Tensor<Dtype>*> &inputs) override {
        std::string device = inputs[0]->device();
        assert(inputs.size()==2 && 
               (inputs[0]->device()==inputs[1]->device()) &&
               inputs[0]->size()==inputs[1]->size());

        NDArray<Dtype> res = NDArray<Dtype>(inputs[0]->shape(),
                                            inputs[0]->strides(),
                                            inputs[0]->offset(),
                                            inputs[0]->device());

        Dtype *a = nullptr, *b = nullptr, *c = nullptr;

        if(device==CPU) {
            a = inputs[0]->_cached_data.cpu();
            b = inputs[1]->_cached_data.cpu();
            c = res.cpu();
            __compute_cpu(a, b, c, inputs[0]->size());
        }

        if(device==CUDA) {
            a = inputs[0]->_cached_data.gpu();
            b = inputs[1]->_cached_data.gpu();
            c = res.gpu();
            __compute_gpu(a, b, c, inputs[0]->size());
        }

        return res;
    }

private:
    void __compute_cpu(Dtype *a, Dtype *b, Dtype *c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
    }

    void __compute_gpu(Dtype *a, Dtype *b, Dtype *c, size_t n) {
        vec_add_wrapper(a, b, c, n);
        cudaDeviceSynchronize(); // Wait for the GPU kernel to finish
    }
};

}

#endif

