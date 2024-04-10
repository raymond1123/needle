#ifndef __EW_OP__
#define __EW_OP__

#include "ops/generic_op.cuh"

template<typename Dtype> class CpuTensor;
template<typename Dtype> class CudaTensor;

constexpr int kBlockSize = 256;
#define NUMWAVES 32

template<typename Dtype>
class ReducedSum {
public:
    __device__ void operator()(size_t n,
                               const Dtype* a, 
                               const Dtype* sum) {

        const int tid = blockIdx.x * kBlockSize + threadIdx.x;
        if (idx < n) atomicAdd(sum, a[tid]);
    }
};

template<typename Dtype>
__global__ void __launch_bounds__(kBlockSize)
ApplyRedSum(size_t n, 
        Dtype* sum, const Dtype* a) {
    //ReducedSum<Dtype> functor = ReducedSum<Dtype>();
    auto functor = ReducedSum<Dtype>();
    functor(n, a, sum);
}

template<typename Dtype>
class SummationOp: public GenericOp<Dtype> {
protected:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

public:
    SummationOp(std::vector<int> axes, std::vector<size_t> shape, OpType op_type):
        GenericOp<Dtype>(op_type), _axes(axes), 
        _reduced_shape(std::vector<size_t>(axes.size())), _shape(shape), _num_blocks(0) {}

    virtual cached_data_type compute(std::vector<cached_data_type> inputs) override {

        assert(inputs.size()==1 && "input number of SummationOp must be 1");

        /* compact inputs */
        for(auto& input: inputs) {
            if(!input->is_compact)
                input->compact();
        }

        cached_data_type tmp_cached = inputs[0];

        for(int i=0; i<_axes.size(); ++i) {

            std::vector<int> pos_axes = _axes;
            int length_shape = _shape.size();

            for(int i=0; i<_axes.size(); ++i) {
                if(_axes[i]<0) pos_axes[i] = length_shape+_axes[i];
                else pos_axes[i] = _axes[i];
                _reduced_shape[i] = pos_axes[i];
            }

        }

        __prepare_pos_axes();
        cached_data_type cached_data = __create_cached_data(_shape,
                                                            inputs[0]->device());

        // base case
        if(_axes.size()==1) {
            _n = inputs[0]->size();

            cudaError_t err = this->_get_num_blocks();
            assert(err==cudaSuccess && "get_num_blocks in SummationOp failed");

            ApplyRedSum<Dtype><<<_num_blocks, kBlockSize, 0>>>(_n,
                                                           cached_data->cached_ptr(), 
                                                           inputs[0]->cached_ptr());
            err = cudaPeekAtLastError();
            assert(err==cudaSuccess && "ApplyRedSum failed");

            cached_data->cached = true;
            cached_data->is_compact = true;

            return cached_data;
        }

        return compute();

    }

    virtual std::vector<cached_data_type> gradient(cached_data_type out_grad, 
                                                   cached_data_type tensor) override {
        return {out_grad};
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
    inline void __prepare_pos_axes() {
        std::vector<int> pos_axes = _axes;
        int length_shape = _shape.size();

        for(int i=0; i<_axes.size(); ++i) {
            if(_axes[i]<0) pos_axes[i] = length_shape+_axes[i];
            else pos_axes[i] = _axes[i];
            _reduced_shape[i] = pos_axes[i];
        }

        _axes = pos_axes;
    }

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
                                                 BackendType device,
                                                 bool create_cache=true) {
        cached_data_type cached_data = nullptr;
        if (device == BackendType::CPU) {
            cached_data.reset(new CpuTensor<Dtype>(shape, create_cache));
        } else if (device == BackendType::CUDA) {
            cached_data.reset(new CudaTensor<Dtype>(shape, create_cache));
        } else {
            throw std::runtime_error("Unsupported backend type.");
        }

        return cached_data;
    }

private:
    size_t _n;
    int _num_blocks;
    std::vector<int> _axes;
    std::vector<size_t> _shape;
    std::vector<size_t> _reduced_shape;
};

#endif

