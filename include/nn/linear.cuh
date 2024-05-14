#ifndef __LINEAR_CUH__
#define __LINEAR_CUH__

#include "common.hpp"
#include "nn/nn_module.cuh"
#include "init/initial.hpp"

namespace py = pybind11;

template<typename Dtype>
class Linear: public Module<Dtype> {

public:
    Linear(int in_features, int out_features, 
           bool bias=true, BackendType device=BackendType::CUDA): 
        Module<Dtype>(), _need_bias(bias),
        _in_features(in_features), _out_features(out_features) {

            std::vector<int32_t> weight_shape = {_out_features, _in_features};
            _weight = kaiming_uniform<Dtype>(weight_shape, device, "relu");

            if(bias) {
                std::vector<int32_t> bias_shape = {1, _out_features};
                _bias = kaiming_uniform<Dtype>(bias_shape, device, "relu");
            }
        }

    void set_params(std::vector<py::array_t<Dtype>>& params,
                    BackendType device=BackendType::CUDA) {
        if(_need_bias) {
            assert(params.size()==2 && "param number of Linear with bias must be 2");
            _bias = Tensor(params[1], device);
        } else 
            assert(params.size()==1 && "param number of Linear without bias must be 1");

        _weight = Tensor(params[0], device);
    }

    virtual std::vector<Tensor<Dtype>> forward(std::vector<Tensor<Dtype>>& tensors) override {
        assert(tensors.size()==1 && "input number of Linear must be 1");

        auto x = tensors[0];
        assert(x.shape()[1]==_in_features &&"shape of input tensor and weight does not match");

        /* y = x@A.T + b */
        auto weight_T = _weight.transpose({0,1});
        auto out = x.matmul(weight_T);

        if(_need_bias) {
            out += _bias.broadcast_to(out.shape());
        }

        return {out};
    }

private:
    Tensor<Dtype> _weight; // shape=(_in_features, _out_features)
    Tensor<Dtype> _bias; // shape = (_out_features, 1)

    bool _need_bias;
    int _in_features;
    int _out_features;
};

#endif

