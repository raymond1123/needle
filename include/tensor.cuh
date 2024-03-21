#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <boost/type_index.hpp>
#include "ops/ops_math.hpp"

namespace py = pybind11;

static int tensor_idx=0;

template<typename Dtype>
class Tensor {
public:
    Tensor(py::array_t<Dtype>& np_array, BackendType backend):
           __backend(backend), __cached(false) {

        // Initialize the backend based on the specified type
        if (__backend == BackendType::CPU) {
            __cached_data = std::make_shared<CpuTensor<Dtype>>(np_array);
        } else if (__backend == BackendType::CUDA) {
            __cached_data = std::make_shared<CudaTensor<Dtype>>(np_array);
        } else {
            throw std::runtime_error("Unsupported backend type.");
        }
        __cached = true;

        #ifdef DEBUG
        tensor_idx++;
        __tensor_idx = tensor_idx;
        __print_tensor_info();
        #endif
    }

    Tensor(BackendType backend, std::shared_ptr<GenericOp<Dtype>> op,
           std::vector<Tensor<Dtype>*>& inputs):
           __backend(backend), _op(op), __inputs(inputs),
           __cached(false) {

        #ifdef DEBUG
        tensor_idx++;
        __tensor_idx = tensor_idx;
        __print_tensor_info();
        #endif
    }

    //TODO move constructor
    /*
    Tensor(Tensor&& other): 
        __backend(other.__backend), 
        __cached_data(other.__cached_data) noexcept {}
    */

    py::array_t<Dtype> to_numpy() {
        return __cached_data->to_numpy();
    }

    std::vector<size_t> shape() {
        return __cached_data->shape();
    }

    /* operations */
    // element-wise addition
    Tensor operator+(Tensor& other) {
        //check same beakend 
        assert(other.__backend == __backend && 
               "backend of operators must be the same");
        std::shared_ptr<GenericOp<Dtype>> op = 
            std::make_shared<AddOp<Dtype>>(__cached_data->size());

        std::vector<Tensor<Dtype>*> inputs;
        inputs.push_back(this);
        inputs.push_back(&other);

        return (*op)(op, inputs);
    }

    static Tensor make_from_op(const std::shared_ptr<GenericOp<Dtype>> op,
                               std::vector<Tensor<Dtype>*>& inputs) {

        assert(inputs.size() > 0 && "number of inputs is zero");

        Tensor new_t = Tensor(inputs[0]->__backend, op, inputs);
        new_t.__cached_data = new_t.realized_cached_data();
        new_t.__cached = true;

        return new_t;
    }

    std::shared_ptr<BaseTensor<Dtype>> realized_cached_data() {
        if(__cached) return __cached_data;

        using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
        std::vector<cached_data_type> cin;
        cin.reserve(__inputs.size());

        for (int i=0; i<__inputs.size(); ++i)
            cin.emplace_back(__inputs[i]->realized_cached_data());

        __cached_data = _op->compute(cin);

        return __cached_data;
    }

    inline BackendType device() {return __backend;}

private:
    void __print_tensor_info() {
        printf("------tensor info---------\n");
        std::vector<int> inputs_idx;
        inputs_idx.reserve(__inputs.size());

        printf("tensor_idx=%d, ", __tensor_idx);

        if(__inputs.size() > 0) {
            printf("inputs_idx= ");
            for(auto &e: __inputs) {
                printf("%d, ", e->__tensor_idx);
            }
        } else {
            printf("inputs_idx=None; ");
        }

        if(_op!=nullptr)
            printf("op: %s\n", boost::typeindex::type_id_runtime(*_op).pretty_name().c_str());
        else
            printf("op: nullptr\n");
        printf("----tensor info done------\n\n");
    }

private:
    int __tensor_idx;
    using generic_op_type = GenericOp<Dtype>;
    std::shared_ptr<generic_op_type> _op;

    BackendType __backend;
    std::shared_ptr<BaseTensor<Dtype>> __cached_data;
    std::vector<Tensor<Dtype>*> __inputs;
    bool __cached;
};

#endif

