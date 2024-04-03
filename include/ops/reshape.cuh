#ifndef __RESHAPE_OP__
#define __RESHAPE_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class ReshapeOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    ReshapeOp(std::vector<size_t> new_shape, OpType op_type):
        GenericOp<Dtype>(op_type), _new_shape(new_shape) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size() == 1 && "number of reshape input must be 1");

        cached_data_type cached_data = inputs[0]->deep_cpy_cached_data();
        cached_data->set_shape(_new_shape);

        return cached_data;
    }

    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor) override {

        cached_data_type out = out_grad->deep_cpy_cached_data();
        out->set_shape(tensor->inputs[0]->shape());

        return {out};
    }

private:
    inline cached_data_type __create_cached_data(const std::vector<size_t>& shape, 
                                                 BackendType device) {
        cached_data_type cached_data = nullptr;
        if (device == BackendType::CPU) {
            cached_data.reset(new CpuTensor<Dtype>(shape));
        } else if (device == BackendType::CUDA) {
            cached_data.reset(new CudaTensor<Dtype>(shape));
        } else {
            throw std::runtime_error("Unsupported backend type.");
        }

        return cached_data;
    }

protected:
    virtual inline cudaError_t _get_num_blocks() override {
        return cudaSuccess;
    }

private:
    std::vector<size_t> _new_shape;
    int _num_blocks;
};

#endif

