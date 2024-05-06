#ifndef __GENERIC_OP__
#define __GENERIC_OP__

#include "common.hpp"
#include "ops/ops_util.cuh"

template<typename Dtype> class Tensor;
template<typename Dtype> class BaseTensor;

template<typename Dtype>
class GenericOp {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    GenericOp(OpType op_type):_op_type(op_type) {};

    virtual cached_data_type compute(std::vector<cached_data_type> inputs)=0;

    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor)=0;

    inline Tensor<Dtype> operator()(const std::shared_ptr<GenericOp<Dtype>> op,
                                    std::vector<cached_data_type>& inputs,
                                    BackendType backend) const {
        return Tensor<Dtype>::make_from_op(op, inputs, backend);
    }

    inline cached_data_type operator()(const std::shared_ptr<GenericOp<Dtype>> op,
                                    std::vector<cached_data_type>& inputs,
                                    BackendType backend, bool op_on_self) const {
        return Tensor<Dtype>::make_from_op_on_self(op, inputs, backend, op_on_self);
    }

    inline int op_type() {return static_cast<int>(_op_type);}

protected:
    virtual inline cudaError_t _get_num_blocks()=0;
public:
    OpType _op_type;
};

template<typename Dtype>
class TupleGenericOp {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    TupleGenericOp(OpType op_type):_op_type(op_type) {};

    virtual cached_data_type compute(std::vector<cached_data_type> inputs)=0;

    virtual std::vector<cached_data_type> gradient(
                            cached_data_type out_grad, 
                            cached_data_type tensor)=0;

    inline Tensor<Dtype> operator()(const std::shared_ptr<TupleGenericOp<Dtype>> op,
                                    std::vector<cached_data_type>& inputs,
                                    BackendType backend) const {
        return Tensor<Dtype>::make_from_op(op, inputs, backend);
    }

    inline cached_data_type operator()(const std::shared_ptr<TupleGenericOp<Dtype>> op,
                                    std::vector<cached_data_type>& inputs,
                                    BackendType backend, bool op_on_self) const {
        return Tensor<Dtype>::make_from_op_on_self(op, inputs, backend, op_on_self);
    }

    inline int op_type() {return static_cast<int>(_op_type);}

protected:
    virtual inline cudaError_t _get_num_blocks()=0;
public:
    OpType _op_type;
};


#endif

///* input one tensor; output one Tensor */
//9.  template<Dtype> Tensor<Dtype> negate(const Tensor<Dtype>& a);
//10. template<Dtype> Tensor<Dtype> log(const Tensor<Dtype>& a);
//11. template<Dtype> Tensor<Dtype> exp(const Tensor<Dtype>& a);
//12. template<Dtype> Tensor<Dtype> relu(const Tensor<Dtype>& a);
//13. template<Dtype> Tensor<Dtype> tanh(const Tensor<Dtype>& a);
//
///* input one tensor and two vector; output one Tensor */
//19. template<Dtype> Tensor<Dtype> dilate(const Tensor<Dtype>& a, const std::vector<int>& axes, const int dilation);
//20. template<Dtype> Tensor<Dtype> undilate(const Tensor<Dtype>& a, const std::vector<int>& axes, const int dilation);
//
///* input un-fixed number of tensors and a vector; output one Tensor */
//21. template<Dtype> Tensor<Dtype> stack(const std::vector<Tensor<Dtype>>, const std::vector<int>& axis);
//
///* input one tensor and a vector; output un-fixed number of tensors */
//22. template<Dtype> std::vector<Tensor<Dtype>> split(const Tensor<Dtype>& a, const std::vector<int>& axis);
//
///* input two tensor and two int; output one Tensor */
//23. template<Dtype> Tensor<Dtype> conv(const Tensor<Dtype>& a, 
//                                  const Tensor<Dtype>& b, 
//                                  const int stride=1, 
//                                  const int padding=1);



