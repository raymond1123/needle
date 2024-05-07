#include "tensor.cuh"
#include "ops/bp/padding.cuh"

template<typename Dtype>
using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

void ccc() {
    std::cout << "ccc function" << std::endl;
}

template<typename Dtype>
Tensor<Dtype> pad(Tensor<Dtype>& tensor, std::vector<int32_t> axes) {
    return tensor.padding(axes);
}

