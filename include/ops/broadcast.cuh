#ifndef __BROADCAST_OP__
#define __BROADCAST_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

template<typename Dtype>
class BroadcastOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    BroadcastOp(std::vector<size_t> new_shape, OpType op_type): 
        GenericOp<Dtype>(op_type), _new_shape(new_shape), 
        _new_strides(std::vector<size_t>(new_shape.size())) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size() == 1 && "number of broadcast input must be 1");

        cached_data_type cached_data = __create_cached_data(_new_shape,
                                                            inputs[0]->device(), 
                                                            false);
        /* without deep cpy data, reuse cached data in inputs[0] */
        cached_data->array = inputs[0]->array;

        __transfer_broadcast_shape(inputs[0]);
        cached_data->set_strides(_new_strides);

        cached_data->cached = true;
        cached_data->is_compact = false;

        return cached_data;
    }

    // TODO gradient using summation
    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor) override {

        cached_data_type out = out_grad->deep_cpy_cached_data();
        //out->set_shape(tensor->inputs[0]->shape());

        return {out};
    }

private:
    void __transfer_broadcast_shape(cached_data_type cached_data) {
        auto org_shape = cached_data->shape();
        auto org_strides = cached_data->strides();

        int size_diff = _new_shape.size() - org_shape.size();
        for(int i=org_shape.size()-1; i>=0; --i) {
            if(org_shape[i]==1 && _new_shape[i+size_diff]>1) continue;
            else _new_strides[i+size_diff] = org_strides[i];
        }
    }

    inline cached_data_type __create_cached_data(const std::vector<size_t>& shape, 
                                                 BackendType device,
                                                 bool create_cache) {
        cached_data_type cached_data = nullptr;
        if (device == BackendType::CPU) {
            cached_data.reset(new CpuTensor<Dtype>(shape, create_cache));
        } else if (device == BackendType::CUDA) {
            cached_data.reset(new CudaTensor<Dtype>(shape, create_cache));
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
    std::vector<size_t> _new_strides;
};

#endif

