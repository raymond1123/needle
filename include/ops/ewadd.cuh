#ifndef __EWADD_OP__
#define __EWADD_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

constexpr int kBlockSize = 256;
#define NUMWAVES 32

template<typename T>
struct AddFunctor {
    __device__ void operator()(int64_t n, T* z, 
                               const T* x, const T* y) const {
        const int tid = blockIdx.x * kBlockSize + threadIdx.x;

        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = x[i] + y[i];
        }
    }
};

/* add operator */
template<typename T>
__global__ void __launch_bounds__(kBlockSize)
ApplyAdd (int64_t n, T* r, const T* a, const T* b) {
    AddFunctor<T> factory;
    factory(n, r, a, b);
}

// R = A + B
template<typename Dtype>
class AddOp: public GenericOp<Dtype> {
//protected:
//    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    AddOp(size_t n): _n(n) {}

    virtual std::shared_ptr<BaseTensor<Dtype>> compute(
                std::vector<std::shared_ptr<BaseTensor<Dtype>>> inputs) override {
        return this->launch_kernel(inputs);
    }

    virtual std::shared_ptr<BaseTensor<Dtype>> launch_kernel(
                std::vector<std::shared_ptr<BaseTensor<Dtype>>> inputs) override {
        assert(inputs.size()==2 && "input number of AddOp is not 2");

        int num_blocks;
        cudaError_t err = this->_get_num_blocks(&num_blocks);
        assert(err==cudaSuccess && "get_num_blocks in AddOp failed");

        std::shared_ptr<BaseTensor<Dtype>> cached_data = __create_cached_data(
                                                             inputs[0]->shape(),
                                                             inputs[0]->device());

        ApplyAdd<Dtype><<<num_blocks, kBlockSize, 0>>>(_n, cached_data->cached_ptr(), 
                                                       inputs[0]->cached_ptr(), 
                                                       inputs[1]->cached_ptr());
        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyAdd failed");

        return cached_data;
    }

    virtual Tensor<Dtype> operator()(const std::shared_ptr< GenericOp<Dtype> > add_op,
                                 std::vector<Tensor<Dtype>*>& inputs) const override {

        return Tensor<Dtype>::make_from_op(add_op, inputs);
    }

protected:
    virtual inline cudaError_t _get_num_blocks(int* num_blocks) override {
        int dev, sm_count, tpm;
        cudaError err = __get_gpu_info(&dev, &sm_count, &tpm);
        *num_blocks = std::max<int>(1, std::min<int64_t>((_n + kBlockSize - 1) / kBlockSize,
                                               sm_count * tpm / kBlockSize * NUMWAVES));
                                               //sm_count * tpm / kBlockSize * kNumWaves));
        return cudaSuccess;
    }

private:
    inline cudaError_t __get_gpu_info(int* dev, int* sm_count, int* tpm) {
        cudaError_t err = cudaGetDevice(dev);
        if (err != cudaSuccess) { return err; }
        err = cudaDeviceGetAttribute(sm_count, cudaDevAttrMultiProcessorCount, *dev);
        if (err != cudaSuccess) { return err; }
        err = cudaDeviceGetAttribute(tpm, cudaDevAttrMaxThreadsPerMultiProcessor, *dev);
        if (err != cudaSuccess) { return err; }
        return cudaSuccess;
    }

    inline std::shared_ptr<BaseTensor<Dtype>> __create_cached_data(
                                                 const std::vector<size_t>& shape, 
                                                 BackendType device) {
        std::shared_ptr<BaseTensor<Dtype>> cached_data = nullptr;
        if (device == BackendType::CPU) {
            cached_data.reset(new CpuTensor<Dtype>(shape));
        } else if (device == BackendType::CUDA) {
            cached_data.reset(new CudaTensor<Dtype>(shape));
        } else {
            throw std::runtime_error("Unsupported backend type.");
        }

        return cached_data;
    }

private:
    size_t _n;
};

#endif

