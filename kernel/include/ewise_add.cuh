#ifndef __EWADD_CUH__
#define __EWADD_CUH__

#include "ops.cuh"

namespace needle {

template <typename Dtype> class Tensor;
template <typename Dtype> class NDArray;
template <typename Dtype> class TensorOP;

template<typename Dtype>
void vec_add_wrapper(Dtype *a, Dtype *b, Dtype *c, size_t n);

template<typename Dtype>
class EwiseAdd: public TensorOP<Dtype> {
public:
    EwiseAdd() = default;
    EwiseAdd(std::string op_name): TensorOP<Dtype>(op_name) {}

    virtual std::shared_ptr<NDArray<Dtype>> compute(const std::vector<const Tensor<Dtype>*> &inputs) override {
        std::string device = inputs[0]->device();
        assert(inputs.size()==2 && 
               (inputs[0]->device()==inputs[1]->device()) &&
               inputs[0]->size()==inputs[1]->size());

        std::shared_ptr<NDArray<Dtype>> res = std::make_shared<NDArray<Dtype>>(
                                            inputs[0]->shape(),
                                            inputs[0]->strides(),
                                            inputs[0]->offset(),
                                            inputs[0]->device());

        Dtype *a = nullptr, *b = nullptr, *c = nullptr;

        if(device==CPU) {
            a = inputs[0]->_cached_data->cpu();
            b = inputs[1]->_cached_data->cpu();
            c = res->cpu();
            __compute_cpu(a, b, c, inputs[0]->size());
        }

        if(device==CUDA) {
            a = inputs[0]->_cached_data->gpu();
            b = inputs[1]->_cached_data->gpu();
            c = res->gpu();
            __compute_gpu(a, b, c, inputs[0]->size());
        }
        return res;
    }

    /*
    virtual std::vector<std::shared_ptr<NDArray<Dtype>*>> gradient(const Tensor<Dtype>* out_grad, 
                                    const Tensor<Dtype>* node) override {

        std::vector<size_t> input0_shape = node->_inputs[0]->shape();
        std::vector<size_t> input1_shape = node->_inputs[0]->shape();

        std::vector<std::shared_ptr<NDArray<Dtype>*>> res[2];

        //res[0] = std::make_shared<NDArray<Dtype>*>();

    }
    */

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

} //namespace needle

#endif

