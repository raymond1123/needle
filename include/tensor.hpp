#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include "common.hpp"
#include "ndarray.hpp"
#include "tensor_op.hpp"

namespace needle {

/*
template <typename Dtype>
class Value {
public:
    Value(const py::list &data,
          const std::string device="cuda"): 
                _cached_data(data, 0, device),
                _device(device) {}

    Value(const NDArray<Dtype> data):
                _cached_data(data),
                _device(data.device()) {}

    Value(const std::vector<const Value<Dtype>*> &inputs,
          std::shared_ptr<TensorOP<Dtype>> op):
                _inputs(inputs),
                _op(op),
                _device(inputs[0]->_device) {}


public:
    NDArray<Dtype> _cached_data;
    const std::vector<const Value<Dtype>*> _inputs;
    std::shared_ptr<TensorOP<Dtype>> _op;
    int _num_outputs;
    std::string _device; // "cpu or cuda"
};
*/

//class Tensor: public Value<Dtype> {
template <typename Dtype>
class Tensor {
public:
    Tensor(const py::list &data,
           const std::string device="cuda"): 
                _cached_data(data, 0, device),
                _device(device) {}

    Tensor(const NDArray<Dtype> data):
                _cached_data(data),
                _device(data.device()) {}

    Tensor(const std::vector<const Tensor<Dtype>*> &inputs,
           std::shared_ptr<TensorOP<Dtype>> op):
                _inputs(inputs),
                _op(op),
                _device(inputs[0]->_device) {}

    virtual ~Tensor() {}

    inline const std::vector<size_t>& shape() const {
        return this->_cached_data.shape();
    }

    inline const std::vector<size_t>& strides() const {
        return this->_cached_data.strides();
    }

    inline size_t offset() const {
        return this->_cached_data.offset();
    }

    inline const std::string& device() const {
        return this->_device;
    }

    inline size_t ndim() const {
        return this->_cached_data.size();
    }

    inline size_t size() const {
        return this->_cached_data.size();
    }

    std::string print() {
        return this->_cached_data.print();
    }

    void backward() {
        compute_gradient_of_variables();
    }

    /* this is actuall DFS */
    const Tensor<Dtype>& realize_cached_data() const {

        // _cached_data is not empty
        if (this->_cached_data.has_data()) {
            return *this;
        }

        auto input_cached_data = [&]() -> const std::vector<const Tensor<Dtype>*> {
            std::vector<const Tensor<Dtype>*> data;
            for(auto &node: this->_inputs) {
                //data.emplace_back(const_cast<Tensor<Dtype>*>(node)->realize_cached_data());
                data.emplace_back(&(node->realize_cached_data()));
            }

            return data;
        };

        this->_cached_data = this->_op->compute(input_cached_data());
        return *this;
    }

    /* operations */
    Tensor<Dtype> operator+(const Tensor<Dtype> &other) {

        std::shared_ptr<TensorOP<Dtype>> op = std::make_shared<EwiseAdd<Dtype>>("EwiseAdd");

        /* 
            NOTE: 
            1. elements in vector are shallowd copied, 
               which means if there is a pointer member in class, 
               the pointers in and out side vector are the same

            2. here cannot use smart pointer, or it will reports segment fault
        */
        const std::vector<const Tensor<Dtype>*> inputs = {this, &other};
        Tensor<Dtype> res = Tensor<Dtype>(inputs, op);

        return res.realize_cached_data();
    }

public:
    mutable NDArray<Dtype> _cached_data;
    const std::vector<const Tensor<Dtype>*> _inputs;
    std::shared_ptr<TensorOP<Dtype>> _op;
    int _num_outputs;
    std::string _device; // "cpu or cuda"

};

} //namespace needle

#endif

