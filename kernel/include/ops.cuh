#ifndef __OPS_CUH__
#define __OPS_CUH__

#include "common.hpp"
#include "tensor.hpp"

namespace needle {

template <typename Dtype> class Tensor;
template <typename Dtype> class NDArray;

template<typename Dtype>
class TensorOP {
public:
    TensorOP() = default;
    TensorOP(std::string op_name): _op_name(op_name) {}

    virtual std::shared_ptr<NDArray<Dtype>> compute(const std::vector<const Tensor<Dtype>*> &inputs)=0;
    /*
    virtual std::vector<std::shared_ptr<NDArray<Dtype>*>> gradient(const Tensor<Dtype>* out_grad, 
                                    const Tensor<Dtype>* node)=0;
                                    */

    //virtual Tensor<Dtype> call(const std::vector<Tensor<Dtype>&> &inputs) {
    //    return Tensor<Dtype>::make_from_op(std::shared_ptr<OP<Dtype>>(this), 
    //                                       inputs);
    //}

public:
    std::string _op_name;
};

} //namespace needle

#endif

