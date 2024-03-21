#ifndef __GENERIC_OP__
#define __GENERIC_OP__

#include "common.hpp"

template<typename Dtype> class Tensor;
template<typename Dtype> class BaseTensor;

template<typename Dtype>
class GenericOp {
//protected:
//    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    GenericOp() {};

    virtual std::shared_ptr<BaseTensor<Dtype>> compute(
                std::vector<std::shared_ptr<BaseTensor<Dtype>>> inputs)=0;
    virtual std::shared_ptr<BaseTensor<Dtype>> launch_kernel(
                std::vector<std::shared_ptr<BaseTensor<Dtype>>> inputs)=0;

    virtual Tensor<Dtype> operator()(const std::shared_ptr<GenericOp<Dtype>> op,
                                   std::vector<Tensor<Dtype>*>& inputs) const=0;

protected:
    virtual inline cudaError_t _get_num_blocks(int* num_blocks)=0;
    std::vector<const Tensor<Dtype>*> _inputs;
};

#endif


///* input two tensor; output one Tensor */
//template<Dtype> Tensor<Dtype> add(const Tensor<Dtype>& a, const Tensor<Dtype>& b);
//template<Dtype> Tensor<Dtype> multiply(const Tensor<Dtype>& a, const Tensor<Dtype>& b);
//template<Dtype> Tensor<Dtype> power(const Tensor<Dtype>& a, const Tensor<Dtype>& b);
//template<Dtype> Tensor<Dtype> divide(const Tensor<Dtype>& a, const Tensor<Dtype>& b);
//
///* input one tensor and a Dtype; output one Tensor */
//template<Dtype> Tensor<Dtype> add_scalar(const Tensor<Dtype>& a, const Dtype scalar);
//template<Dtype> Tensor<Dtype> mul_scalar(const Tensor<Dtype>& a, const Dtype scalar);
//template<Dtype> Tensor<Dtype> power_scalar(const Tensor<Dtype>& a, const Dtype scalar);
//template<Dtype> Tensor<Dtype> divide_scalar(const Tensor<Dtype>& a, const Dtype scalar);
//template<Dtype> Tensor<Dtype> matmul(const Tensor<Dtype>& a, const Tensor<Dtype>& b);
//
///* input one tensor; output one Tensor */
//template<Dtype> Tensor<Dtype> negate(const Tensor<Dtype>& a);
//template<Dtype> Tensor<Dtype> log(const Tensor<Dtype>& a);
//template<Dtype> Tensor<Dtype> exp(const Tensor<Dtype>& a);
//template<Dtype> Tensor<Dtype> relu(const Tensor<Dtype>& a);
//template<Dtype> Tensor<Dtype> tanh(const Tensor<Dtype>& a);
//
///* input one tensor and a vector; output one Tensor */
//template<Dtype> Tensor<Dtype> reshape(const Tensor<Dtype>& a, const std::vector<int>& shape);
//template<Dtype> Tensor<Dtype> broadcast_to(const Tensor<Dtype>& a, const std::vector<int>& shape);
//template<Dtype> Tensor<Dtype> transpose(const Tensor<Dtype>& a, const std::vector<int>& axes);
//template<Dtype> Tensor<Dtype> summation(const Tensor<Dtype>& a, const std::vector<int>& axes);
//template<Dtype> Tensor<Dtype> flip(const Tensor<Dtype>&a, const std::vector<int>& axes);
//
///* input one tensor and two vector; output one Tensor */
//template<Dtype> Tensor<Dtype> dilate(const Tensor<Dtype>& a, const std::vector<int>& axes, const int dilation);
//template<Dtype> Tensor<Dtype> undilate(const Tensor<Dtype>& a, const std::vector<int>& axes, const int dilation);
//
///* input un-fixed number of tensors and a vector; output one Tensor */
//template<Dtype> Tensor<Dtype> stack(const std::vector<Tensor<Dtype>>, const std::vector<int>& axis);
//
///* input one tensor and a vector; output un-fixed number of tensors */
//template<Dtype> std::vector<Tensor<Dtype>> split(const Tensor<Dtype>& a, const std::vector<int>& axis);
//
///* input two tensor and two int; output one Tensor */
//template<Dtype> Tensor<Dtype> conv(const Tensor<Dtype>& a, 
//                                   const Tensor<Dtype>& b, 
//                                   const int stride=1, 
//                                   const int padding=1);



