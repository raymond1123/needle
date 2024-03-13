#ifndef __TENSOR_OP__
#define __TENSOR_OP__

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

template<typename FunctorT, typename R, typename... IN>
__global__ void __launch_bounds__(kBlockSize)
ApplyGeneric(FunctorT factory, int64_t n, R r, const IN... in) {
  factory(n, r, in...);
}

template<typename FunctorT, typename R, typename... IN>
class OP {
public:
  cudaError_t Launch(FunctorT factory, int64_t n, R r, const IN... in) {
    return _launch_kernel(factory, n, r, in...);
  }

protected:
  cudaError_t _launch_kernel(FunctorT factory, int64_t n, R r, const IN... in) {
    int num_blocks;
    cudaError_t err = _get_num_blocks(n, &num_blocks);
    if (err != cudaSuccess) { return err; }

    ApplyGeneric<FunctorT, R, IN...><<<num_blocks, kBlockSize, 0>>>(
        factory, n, r, in...);

    return cudaPeekAtLastError();
  }

  virtual inline cudaError_t _get_num_blocks(int64_t n, int* num_blocks) {
    int dev, sm_count, tpm;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
    err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
    err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
    *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                     sm_count * tpm / kBlockSize * kNumWaves));
    return cudaSuccess;
  }
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



