#ifndef __TENSOR_OP__
#define __TENSOR_OP__

#include "ops/tensor_op.cuh"

template<typename FunctorT, typename R, typename A, typename B>
class EWiseMul: GenericLauncher<FunctorT, R, A, B>{
public:
  cudaError_t compute(FunctorT functor, int64_t n, R r, 
                      const A a, const B b) {
    return this->Launch(functor, n, r, a, b);
  }

private:
  virtual inline cudaError_t _get_num_blocks(int64_t n, int* num_blocks) override {
      printf("override _get_num_blocks\n");
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

template<typename T1, typename T2, typename T3>
struct MultiplyFunctor {
  __device__ void operator()(int64_t n, T1 z, 
                             const T2 x, const T3 y) const {
    const int tid = blockIdx.x * kBlockSize + threadIdx.x;

    for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
      z[i] = x[i] + y[i];
    }
  }
};

/* usage
    // elementwise template
    half a_host = 1;
    using func_type = MultiplyFunctor<half*, const half*, const half*, const half>;
    EWiseMul<func_type, half*, const half*, const half*, const half> ewmul;

    ewmul.compute(MultiplyFunctor<half*, const half*, const half*, const half>(), 
                  N, output_device, x_device, y_device, a_host);

    cudaMemcpy(output_host, output_device, N * sizeof(half), cudaMemcpyDeviceToHost);
*/


#endif

