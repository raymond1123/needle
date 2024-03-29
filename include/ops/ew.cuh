#ifndef __EW_OP__
#define __EW_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

constexpr int kBlockSize = 256;
#define NUMWAVES 32

template<typename Dtype>
class EWAddTensor {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype* y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = x[i] + y[i];
        }
    }
};

template<typename Dtype>
class EWAddScalar {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = x[i] + y;
        }
    }
};

template<typename Dtype>
class EWMinusTensor {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype* y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = x[i] - y[i];
        }
    }
};

template<typename Dtype>
class EWMinusScalar {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = x[i] - y;
        }
    }
};

template<typename Dtype>
class EWMulTensor {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype* y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = x[i] * y[i];
        }
    }
};

template<typename Dtype>
class EWMulScalar {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = x[i] * y;
        }
    }
};

template<typename Dtype>
class EWDivTensor {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype* y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = x[i] / y[i];
        }
    }
};

template<typename Dtype>
class EWDivScalar {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = x[i] / y;
        }
    }
};

template<typename Dtype>
class EWPowTensor {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype* y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = pow(x[i], y[i]);
        }
    }
};

template<typename Dtype>
class EWPowScalar {
public:
    __device__ void operator()(int64_t n, Dtype* z, 
                               const Dtype* x, 
                               const Dtype y) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
          z[i] = pow(x[i], y);
        }
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyEW(OpType op_type, int64_t n, 
        Dtype* r, const Dtype* a, const Dtype* b) {
    if(op_type==OpType::EWAddTensor) {
        EWAddTensor<Dtype> functor = EWAddTensor<Dtype>();
        functor(n, r, a, b);
    } else if(op_type==OpType::EWMinusTensor) {
        EWMinusTensor<Dtype> functor = EWMinusTensor<Dtype>();
        functor(n, r, a, b);
    } else if(op_type==OpType::EWMulTensor) {
        EWMulTensor<Dtype> functor = EWMulTensor<Dtype>();
        functor(n, r, a, b);
    } else if(op_type==OpType::EWDivTensor) {
        EWDivTensor<Dtype> functor = EWDivTensor<Dtype>();
        functor(n, r, a, b);
    } else if(op_type==OpType::EWPowTensor) {
        EWPowTensor<Dtype> functor = EWPowTensor<Dtype>();
        functor(n, r, a, b);
    }
}

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyEW(OpType op_type, int64_t n, 
        Dtype* r, const Dtype* a, const Dtype b) {
    if(op_type==OpType::EWAddScalar) {
        EWAddScalar<Dtype> functor = EWAddScalar<Dtype>();
        functor(n, r, a, b);
    } else if(op_type==OpType::EWMinusScalar) {
        EWMinusScalar<Dtype> functor = EWMinusScalar<Dtype>();
        functor(n, r, a, b);
    } else if(op_type==OpType::EWMulScalar) {
        EWMulScalar<Dtype> functor = EWMulScalar<Dtype>();
        functor(n, r, a, b);
    } else if(op_type==OpType::EWDivScalar) {
        EWDivScalar<Dtype> functor = EWDivScalar<Dtype>();
        functor(n, r, a, b);
    } else if(op_type==OpType::EWPowScalar) {
        EWPowScalar<Dtype> functor = EWPowScalar<Dtype>();
        functor(n, r, a, b);
    }
}

// R = A + B
template<typename Dtype>
class EWOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    EWOp(size_t n, OpType op_type, Dtype scalar=0):
        GenericOp<Dtype>(op_type),
        _n(n), _scalar(scalar), _num_blocks(0) {}

    virtual cached_data_type compute(
                std::vector<cached_data_type> inputs) override {

        int num_inputs = inputs.size();
        assert((num_inputs==2 || num_inputs==1) && "input number of EWOp must be 1 or 2");

        cudaError_t err = this->_get_num_blocks();
        assert(err==cudaSuccess && "get_num_blocks in EWOp failed");

        cached_data_type cached_data = __create_cached_data(inputs[0]->shape(),
                                                            inputs[0]->device());

        // TODO use function<> here, not if else in kernel function
        if (num_inputs==2) {
            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(this->_op_type, _n,
                                                       cached_data->cached_ptr(), 
                                                       inputs[0]->cached_ptr(), 
                                                       inputs[1]->cached_ptr());
        } else {
            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(this->_op_type, _n,
                                                       cached_data->cached_ptr(), 
                                                       inputs[0]->cached_ptr(), 
                                                       _scalar);
        }

        err = cudaPeekAtLastError();
        assert(err==cudaSuccess && "ApplyEW failed");

        return cached_data;
    }

    virtual std::vector<std::shared_ptr<Tensor<Dtype>>> gradient(
                            std::shared_ptr<Tensor<Dtype>> out_grad, 
                            Tensor<Dtype>* tensor) override {
        if(_num_blocks==0) {
            cudaError_t err = this->_get_num_blocks();
        }

        if(this->_op_type == OpType::EWAddTensor) {
            return {out_grad, out_grad};
        } else if(this->_op_type == OpType::EWAddScalar) {
            return {out_grad};
        } else if(this->_op_type == OpType::EWMinusTensor) {
            auto out1 = out_grad;
            auto out2 = std::make_shared<Tensor<Dtype>>(out_grad->device());
            auto cached_data = __create_cached_data(out_grad->shape(),
                                                    out_grad->device());
            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWMulScalar, _n,
                                                       cached_data->cached_ptr(), 
                                                       out_grad->cached_ptr(), 
                                                       static_cast<Dtype>(-1));
            out2->reset_cached_data(cached_data);

            return {out1, out2};
        } else if(this->_op_type == OpType::EWMinusScalar) {
            return {out_grad};
        } else if(this->_op_type == OpType::EWMulTensor) {

            auto inputs = tensor->get_inputs();

            auto out1 = std::make_shared<Tensor<Dtype>>(out_grad->device());
            auto out2 = std::make_shared<Tensor<Dtype>>(out_grad->device());

            auto cached_data_1 = __create_cached_data(out_grad->shape(),
                                                       out_grad->device());
            auto cached_data_2 = __create_cached_data(out_grad->shape(),
                                                       out_grad->device());

            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWMulTensor, _n,
                                                       cached_data_1->cached_ptr(), 
                                                       out_grad->cached_ptr(), 
                                                       inputs[1]->cached_ptr());

            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWMulTensor, _n,
                                                       cached_data_2->cached_ptr(), 
                                                       out_grad->cached_ptr(), 
                                                       inputs[0]->cached_ptr());

            out1->reset_cached_data(cached_data_1);
            out2->reset_cached_data(cached_data_2);

            return {out1, out2};

        } else if(this->_op_type == OpType::EWMulScalar) {
            auto out = std::make_shared<Tensor<Dtype>>(out_grad->device());
            auto cached_data = __create_cached_data(out_grad->shape(),
                                                       out_grad->device());
            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWMulScalar, _n,
                                                       cached_data->cached_ptr(), 
                                                       out_grad->cached_ptr(), 
                                                       _scalar);
            out->reset_cached_data(cached_data);

            return {out};

        } else if(this->_op_type == OpType::EWDivTensor) {
            auto inputs = tensor->get_inputs();

            auto out1 = std::make_shared<Tensor<Dtype>>(out_grad->device());
            auto out2 = std::make_shared<Tensor<Dtype>>(out_grad->device());

            auto cached_data_1 = __create_cached_data(out_grad->shape(),
                                                       out_grad->device());
            auto cached_data_2 = __create_cached_data(out_grad->shape(),
                                                       out_grad->device());

            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWPowScalar, _n,
                                                       cached_data_1->cached_ptr(), 
                                                       inputs[1]->cached_ptr(),
                                                       static_cast<Dtype>(-1));

            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWMulTensor, _n,
                                                       cached_data_1->cached_ptr(), 
                                                       out_grad->cached_ptr(), 
                                                       cached_data_1->cached_ptr());

            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWPowScalar, _n,
                                                       cached_data_2->cached_ptr(), 
                                                       inputs[1]->cached_ptr(),
                                                       static_cast<Dtype>(-2));

            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWMulTensor, _n,
                                                       cached_data_2->cached_ptr(), 
                                                       inputs[0]->cached_ptr(), 
                                                       cached_data_2->cached_ptr());

            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWMulTensor, _n,
                                                       cached_data_2->cached_ptr(), 
                                                       out_grad->cached_ptr(), 
                                                       cached_data_2->cached_ptr());


            out1->reset_cached_data(cached_data_1);
            out2->reset_cached_data(cached_data_2);

            return {out1, out2};

        } else if(this->_op_type == OpType::EWDivScalar) {
            auto out = std::make_shared<Tensor<Dtype>>(out_grad->device());
            auto cached_data = __create_cached_data(out_grad->shape(),
                                                       out_grad->device());
            ApplyEW<Dtype><<<_num_blocks, kBlockSize, 0>>>(OpType::EWMulScalar, _n,
                                                       cached_data->cached_ptr(), 
                                                       out_grad->cached_ptr(), 
                                                       static_cast<Dtype>(1/_scalar));
            out->reset_cached_data(cached_data);

            return {out};
        }

        return {};
    }

    virtual Tensor<Dtype> operator()(const std::shared_ptr< GenericOp<Dtype> > add_op,
                                 std::vector<Tensor<Dtype>*>& inputs) const override {

        return Tensor<Dtype>::make_from_op(add_op, inputs);
    }

protected:
    virtual inline cudaError_t _get_num_blocks() override {
        int dev, sm_count, tpm;
        cudaError err = __get_gpu_info(&dev, &sm_count, &tpm);
        _num_blocks = std::max<int>(1, std::min<int64_t>((_n + kBlockSize - 1) / kBlockSize,
                                               sm_count * tpm / kBlockSize * NUMWAVES));
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

    inline cached_data_type __create_cached_data(const std::vector<size_t>& shape, 
                                                 BackendType device) {
        cached_data_type cached_data = nullptr;
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
    int _num_blocks;
    Dtype _scalar;
};

#endif

