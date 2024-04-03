#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

//#include <boost/type_index.hpp>
#include "ops/ops_math.hpp"

namespace py = pybind11;

static int tensor_idx=0;

template<typename Dtype>
class Tensor {
public:
    /* constructor */
    Tensor(py::array_t<Dtype>& np_array, BackendType backend);
    //Tensor(std::vector<size_t> &shape, BackendType backend);
    Tensor(BackendType backend, 
           std::shared_ptr<GenericOp<Dtype>> op=nullptr,
           std::vector<Tensor<Dtype>*> inputs={nullptr});

    /* debug api */
    inline int idx() {return __tensor_idx;}
    inline std::vector<int> inputs_idx() {
        std::vector<int> idxs;
        for(auto& in: __inputs) {
            idxs.push_back(in->__tensor_idx);
        }

        return idxs;
    }
    /* debug api done */

    static Tensor ones(std::vector<size_t> shape, BackendType backend);
    static Tensor zeros(std::vector<size_t> shape, BackendType backend);

    /* move/cpy constructor */
    Tensor(Tensor&& other) noexcept;
    Tensor(Tensor& other);

    /* move/cpy operator= */
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor& operator=(Tensor& other);
    std::shared_ptr<BaseTensor<Dtype>> deep_cpy_cached_data();

    inline py::array_t<Dtype> to_numpy() { return __cached_data->to_numpy(); }
    inline py::array_t<Dtype> grad() { return __grad->__cached_data->to_numpy(); }
    inline std::vector<size_t> shape() { return __cached_data->shape(); }
    inline std::vector<size_t> strides() { return __cached_data->strides(); }
    inline BackendType device() {return __backend;}

    inline Dtype* cached_ptr() {return __cached_data->cached_ptr();}
    inline std::vector<Tensor<Dtype>*> get_inputs() {
        return __inputs;
    }

    inline void reset_cached_data(std::shared_ptr<BaseTensor<Dtype>> cached_data) {
        __cached_data = cached_data;
    }

    static Tensor make_from_op(const std::shared_ptr<GenericOp<Dtype>> op,
                               std::vector<Tensor<Dtype>*>& inputs);

    std::shared_ptr<BaseTensor<Dtype>> realized_cached_data();

    /* getitem */
    Tensor operator[](std::vector<size_t> indices);

    /* operations */
    // element-wise addition
    Tensor operator+(Tensor& other);
    Tensor operator+(const Dtype scalar);
    //Tensor operator+=(const Tensor& other);
    Tensor operator-(Tensor& other);
    Tensor operator-(const Dtype scalar);
    Tensor operator*(Tensor& other);
    Tensor operator*(const Dtype scalar);
    Tensor operator/(Tensor& other);
    Tensor operator/(const Dtype scalar);
    Tensor op_pow(Tensor& other);
    Tensor op_pow(const Dtype scalar);


    Tensor reshape(std::vector<size_t> new_shape);
    //Tensor broadcast_to();

    /* backward */
    void backward();
    //void backward(std::shared_ptr<Tensor> out_grad);

private:
    void __print_tensor_info(std::string ctor_type);
    void __compute_gradient(Tensor<Dtype>* out_tensor, 
                            std::shared_ptr<Tensor<Dtype>> out_grad);

    std::shared_ptr<Tensor<Dtype>> __sum_grad(
                std::vector<std::shared_ptr<Tensor<Dtype>>>& input_grads);

    /* DFS for graph */
    void __find_topo_sort(std::vector<Tensor<Dtype>*> tensors,
                          std::vector<Tensor<Dtype>*>& reverse_topo_order);

    void __topo_sort_dfs(Tensor<Dtype>* tensor_ptr, 
                std::unordered_map<Tensor<Dtype>*, bool>& visited,
                std::vector<Tensor<Dtype>*>& topo_order);


private:
    int __tensor_idx;
    BackendType __backend;
    bool __cached;

    std::shared_ptr<GenericOp<Dtype>> _op;

    std::shared_ptr<BaseTensor<Dtype>> __cached_data;
    std::vector<Tensor<Dtype>*> __inputs;

    std::shared_ptr<Tensor<Dtype>> __grad;
};

template<typename Dtype>
Tensor<Dtype>::Tensor(py::array_t<Dtype>& np_array, BackendType backend):
       __backend(backend), __cached(false) {

    // Initialize the backend based on the specified type
    if (__backend == BackendType::CPU) {
        __cached_data = std::make_shared<CpuTensor<Dtype>>(np_array);
    } else if (__backend == BackendType::CUDA) {
        __cached_data = std::make_shared<CudaTensor<Dtype>>(np_array);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }
    __cached = true;

    #ifdef DEBUG
    tensor_idx++;
    __tensor_idx = tensor_idx;
    __print_tensor_info("plane ctor");
    #endif
}

template<typename Dtype>
Tensor<Dtype>::Tensor(BackendType backend, 
           std::shared_ptr<GenericOp<Dtype>> op,
           std::vector<Tensor<Dtype>*> inputs):
           __backend(backend), _op(op), __inputs(inputs),
           __cached(false) {

    #ifdef DEBUG
    tensor_idx++;
    __tensor_idx = tensor_idx;
    __print_tensor_info("ctor");
    #endif
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::ones(std::vector<size_t> shape,
                                  BackendType backend) {
    Tensor<Dtype> tensor = Tensor<Dtype>(backend);

    if (backend == BackendType::CPU) {
        tensor.__cached_data = std::make_shared<CpuTensor<Dtype>>(shape);
    } else if (backend == BackendType::CUDA) {
        tensor.__cached_data = std::make_shared<CudaTensor<Dtype>>(shape);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    tensor.__cached_data->ones();
    tensor.__cached = true;

    #ifdef DEBUG
    tensor_idx++;
    tensor.__tensor_idx = tensor_idx;
    tensor.__print_tensor_info("ones");
    #endif

    return tensor;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::zeros(std::vector<size_t> shape,
                                  BackendType backend) {
    Tensor<Dtype> tensor = Tensor<Dtype>(backend);

    if (backend == BackendType::CPU) {
        tensor.__cached_data = std::make_shared<CpuTensor<Dtype>>(shape);
    } else if (backend == BackendType::CUDA) {
        tensor.__cached_data = std::make_shared<CudaTensor<Dtype>>(shape);
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    tensor.__cached_data->zeros();
    tensor.__cached = true;

    #ifdef DEBUG
    tensor_idx++;
    tensor.__tensor_idx = tensor_idx;
    tensor.__print_tensor_info("zeros");
    #endif

    return tensor;
}

template<typename Dtype>
Tensor<Dtype>::Tensor(Tensor&& other) noexcept:
        __tensor_idx(other.__tensor_idx), __backend(other.__backend), 
        _op(other._op), __cached_data(other.__cached_data),
        __inputs(other.__inputs), __cached(other.__cached) {

    other._op = nullptr;
    other.__cached_data = nullptr;
    for(auto& input: other.__inputs)
        input = nullptr;

    #ifdef DEBUG
    printf("tensor_idx:%d, original=%p, new=%p, move constructor\n", 
           __tensor_idx, &other, this);
    __print_tensor_info("move constructor");
    #endif
}

template<typename Dtype>
Tensor<Dtype>& Tensor<Dtype>::operator=(Tensor<Dtype>&& other) noexcept {

    if(this == &other) return *this;

    __tensor_idx = other.__tensor_idx;
    __backend = other.__backend;
    _op = other._op;
    __cached_data = other.__cached_data;
    __inputs = other.__inputs;
    __cached = other.__cached;

    other._op = nullptr;
    other.__cached_data = nullptr;
    for(auto& input: other.__inputs)
        input = nullptr;

    #ifdef DEBUG
    printf("tensor_idx:%d, move operator=\n", __tensor_idx);
    #endif

    return *this;
}

template<typename Dtype>
Tensor<Dtype>::Tensor(Tensor& other): __backend(other.__backend), 
                                      __cached(other.__cached) {

    __inputs = {nullptr};
    _op = nullptr;

    __cached_data = other.__cached_data->deep_cpy_cached_data();

    #ifdef DEBUG
    tensor_idx++;
    __tensor_idx = tensor_idx;
    __print_tensor_info("cpy ctor");
    printf("tensor_idx:%d, cpy constructor\n", __tensor_idx);
    #endif
}

template<typename Dtype>
Tensor<Dtype>& Tensor<Dtype>::operator=(Tensor<Dtype>& other) {
    if(this==&other) return *this;

    __backend = other.__backend;
    __cached = other.__cached;

    __inputs = {nullptr};
    _op = nullptr;

    __cached_data = other.__cached_data->deep_cpy_cached_data();


    #ifdef DEBUG
    tensor_idx++;
    __tensor_idx = tensor_idx;
    __print_tensor_info("cpy operator=");
    printf("tensor_idx:%d, cpy operator=\n", __tensor_idx);
    #endif

    return *this;
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> Tensor<Dtype>::deep_cpy_cached_data() {
    return __cached_data->deep_cpy_cached_data();
}

// this = this + other
//template<typename Dtype>
//Tensor<Dtype> Tensor<Dtype>::operator+=(const Tensor<Dtype>& other) {
//    //check same beakend 
//    assert(other.__backend == __backend && 
//           "backend of operators must be the same");
//
//    /*
//    std::shared_ptr<EWOp<Dtype>> op =
//                 std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
//                                               OpType::EWAddTensor);
//
//    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
//    std::vector<cached_data_type> cin = {__cached_data, other.__cached_data};
//
//    __cached_data = op->compute(cin);
//
//    return *this;
//    */
//    printf("ooooooooooooooooperators+=\n");
//
//    return (*this)+other;
//}

// return = this + other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator+(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.__backend == __backend && 
           "backend of operators must be the same");
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWAddTensor);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    inputs.push_back(&other);
    printf("===============+\n");

    return (*op)(op, inputs);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator+(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWAddScalar,
                                      scalar);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    printf("===============+\n");

    return (*op)(op, inputs);
}

// return = this - other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator-(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.__backend ==__backend && 
           "backend of operators must be the same");
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWMinusTensor);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    inputs.push_back(&other);
    printf("===============+\n");

    return (*op)(op, inputs);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator-(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWMinusScalar,
                                      scalar);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    printf("===============+\n");

    return (*op)(op, inputs);
}

// return = this * other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator*(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.__backend ==__backend && 
           "backend of operators must be the same");
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWMulTensor);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    inputs.push_back(&other);
    printf("===============+\n");

    return (*op)(op, inputs);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator*(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWMulScalar,
                                      scalar);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    printf("===============+\n");

    return (*op)(op, inputs);
}

// return = this * other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator/(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.__backend ==__backend && 
           "backend of operators must be the same");
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWDivTensor);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    inputs.push_back(&other);
    printf("===============+\n");

    return (*op)(op, inputs);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator/(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWDivScalar,
                                      scalar);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    printf("===============+\n");

    return (*op)(op, inputs);
}

// return = this * other
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::op_pow(Tensor<Dtype>& other) {
    //check same beakend 
    assert(other.__backend ==__backend && 
           "backend of operators must be the same");
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWPowTensor);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    inputs.push_back(&other);
    printf("===============+\n");

    return (*op)(op, inputs);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::op_pow(const Dtype scalar) {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::EWPowScalar,
                                      scalar);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    printf("===============+\n");

    return (*op)(op, inputs);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::reshape(std::vector<size_t> new_shape) {
    size_t new_size = 1;
    for (auto &s: new_shape)
        new_size *= s;
    assert(new_size==__cached_data->size() && "reshape input new_shape is not legal");

    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<ReshapeOp<Dtype>>(new_shape, OpType::Reshape);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    printf("===============+\n");

    return (*op)(op, inputs);
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator[](std::vector<size_t> indices) {

}

/*
template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::broadcast_to() {
    std::shared_ptr<GenericOp<Dtype>> op = 
        std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                      OpType::BroadcastTo,
                                      scalar);

    std::vector<Tensor<Dtype>*> inputs;
    inputs.push_back(this);
    printf("===============+\n");

    return (*op)(op, inputs);
}
*/

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::make_from_op(const std::shared_ptr<GenericOp<Dtype>> op,
                                          std::vector<Tensor<Dtype>*>& inputs) {
    assert(inputs.size() > 0 && "number of inputs is zero");

    Tensor<Dtype> new_t = Tensor<Dtype>(inputs[0]->__backend, op, inputs);
    new_t.__cached_data = new_t.realized_cached_data();
    new_t.__cached = true;

    return new_t;
}

/*
template<typename Dtype>
void Tensor<Dtype>::backward(std::shared_ptr<Tensor<Dtype>> out_grad) {
    // if leaf tensor, create its own ones tensor 
    if(out_grad==nullptr) {
        out_grad = std::make_shared<Tensor<Dtype>>(__backend);

        if (__backend == BackendType::CPU) {
            out_grad->__cached_data = 
                        std::make_shared<CpuTensor<Dtype>>(__cached_data->shape());
        } else if (__backend == BackendType::CUDA) {
            out_grad->__cached_data = 
                        std::make_shared<CudaTensor<Dtype>>(__cached_data->shape());
        } else {
            throw std::runtime_error("Unsupported backend type.");
        }

        out_grad->__cached_data->ones();
    }

    __compute_gradient(this, out_grad);
}
*/

template<typename Dtype>
void Tensor<Dtype>::backward() {
    __print_tensor_info("backward");
    std::shared_ptr<Tensor<Dtype>> out_grad = 
        std::make_shared<Tensor<Dtype>>(__backend);

    if (__backend == BackendType::CPU) {
        out_grad->__cached_data = 
                    std::make_shared<CpuTensor<Dtype>>(__cached_data->shape());
    } else if (__backend == BackendType::CUDA) {
        out_grad->__cached_data = 
                    std::make_shared<CudaTensor<Dtype>>(__cached_data->shape());
    } else {
        throw std::runtime_error("Unsupported backend type.");
    }

    out_grad->__cached_data->ones();

    __compute_gradient(this, out_grad);
}

template<typename Dtype>
void Tensor<Dtype>::__compute_gradient(Tensor<Dtype>* out_tensor, 
                                       std::shared_ptr<Tensor<Dtype>> out_grad) {

    using grad_type = std::shared_ptr<Tensor<Dtype>>;
    std::unordered_map<Tensor<Dtype>*, std::vector<grad_type>> grad_map;

    grad_map[out_tensor] = {out_grad};
    std::vector<Tensor<Dtype>*> reverse_topo_order;

    __find_topo_sort({out_tensor}, reverse_topo_order);
    std::reverse(reverse_topo_order.begin(), reverse_topo_order.end());

    #ifdef DEBUG
    for(auto& tensor: reverse_topo_order) {
        int op_type = -1;
        if (tensor->_op != nullptr) {
            op_type = tensor->_op->op_type();
        } 

        printf("tensor_idx=%d, op=%d, addr=%p-->", 
               tensor->__tensor_idx, op_type, tensor);
    }
    printf("\n");
    #endif

    int cnt = 0;
    for(auto& tensor: reverse_topo_order) {
        tensor->__grad = __sum_grad(grad_map[tensor]);

        if(tensor->_op!=nullptr) {
            std::vector<grad_type> grads;

            #ifdef DEBUG
            printf("cnt=%d:",cnt);
            tensor->__print_tensor_info("__ccccccccompute_gradient");
            #endif

            grads = tensor->_op->gradient(tensor->__grad, tensor);

            for(int i=0; i<tensor->__inputs.size(); ++i) {
                grad_map[tensor->__inputs[i]].push_back(grads[i]);
            }
        }
        cnt += 1;
    }
}

template<typename Dtype>
std::shared_ptr<Tensor<Dtype>> Tensor<Dtype>::__sum_grad(std::vector<std::shared_ptr<Tensor<Dtype>>>& input_grads) {
    assert(input_grads.size() > 0 && "at least one input gradients");

    // create grad Tensor
    std::shared_ptr<Tensor<Dtype>> grad = std::make_shared<Tensor<Dtype>>(__backend);
    if (__backend == BackendType::CPU) {
        grad->__cached_data = 
                  std::make_shared<CpuTensor<Dtype>>(input_grads[0]->__cached_data->shape());
    } else if (__backend == BackendType::CUDA) {
        grad->__cached_data = 
                  std::make_shared<CudaTensor<Dtype>>(input_grads[0]->__cached_data->shape());
    }

    grad->__cached_data->zeros();

    for(auto &in_grad: input_grads) {
        //(*grad) += (*in_grad);
        std::shared_ptr<EWOp<Dtype>> op =
                 std::make_shared<EWOp<Dtype>>(__cached_data->size(), 
                                               OpType::EWAddTensor, 0, true);
        using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
        std::vector<cached_data_type> cin = {grad->__cached_data, in_grad->__cached_data};

        op->compute(cin);
    }

    return grad;
}

template<typename Dtype>
void Tensor<Dtype>::__find_topo_sort(std::vector<Tensor<Dtype>*> tensors,
                      std::vector<Tensor<Dtype>*>& reverse_topo_order) {
    std::unordered_map<Tensor<Dtype>*, bool> visited;

    for (auto& tensor_ptr: tensors) {
        if(!visited[tensor_ptr])
            __topo_sort_dfs(tensor_ptr, visited, reverse_topo_order);
    }
}

template<typename Dtype>
void Tensor<Dtype>::__topo_sort_dfs(Tensor<Dtype>* tensor_ptr, 
                std::unordered_map<Tensor<Dtype>*, bool>& visited,
                std::vector<Tensor<Dtype>*>& reverse_topo_order) {
    visited[tensor_ptr] = true;
    for(auto& input: tensor_ptr->__inputs) {
        if(!visited[input]) {
            visited[input] = true;
            __topo_sort_dfs(input, visited, reverse_topo_order);
        }
    }

    reverse_topo_order.push_back(tensor_ptr);
    printf("zzzzzzzzzzzzzzzzzzzz: reverse_topo_order.size()=%lu\n", reverse_topo_order.size());
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> Tensor<Dtype>::realized_cached_data() {
    if(__cached) return __cached_data;

    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
    std::vector<cached_data_type> cin;
    cin.reserve(__inputs.size());

    for (int i=0; i<__inputs.size(); ++i)
        cin.emplace_back(__inputs[i]->realized_cached_data());

    __cached_data = _op->compute(cin);

    return __cached_data;
}


template<typename Dtype>
void Tensor<Dtype>::__print_tensor_info(std::string ctor_type) {
    printf("------tensor info---------\n");
    printf("%s:", ctor_type.c_str());
    std::vector<int> inputs_idx;
    inputs_idx.reserve(__inputs.size());

    printf("tensor_idx=%d; ", __tensor_idx);
    printf("tensor_addr=%p; ", this);

    if(__inputs.size() > 0) {
        printf("inputs_idx= ");
        for(auto &e: __inputs) {
            if(e!=nullptr)
                printf("%d, ", e->__tensor_idx);
        }

        printf("; inputs_addr= ");
        for(auto &e: __inputs) {
            printf("%p, ", e);
        }

    } else {
        printf("inputs_idx=None; ");
    }

    if(_op!=nullptr)
        printf("op: %d\n", static_cast<int>(_op->_op_type));
    else
        printf("op: nullptr\n");
    printf("----tensor info done------\n\n");
}

#endif

