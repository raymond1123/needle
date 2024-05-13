#ifndef __NN_BASIC_CUH__
#define __NN_BASIC_CUH__

#include "tensor.cuh"
#include "ops/bp/padding.cuh"

template<typename Dtype>
using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

/* special Tensor that represents parameters */
template<typename Dtype>
class Parameter: public Tensor<Dtype> {
};

template<typename Dtype>
class Module {
public:
    using module_type = std::shared_ptr<Module<Dtype>>;

    Module(): _training(true) {}
    Module(std::vector<module_type>& modules): 
        _training(true), _sub_modules(modules) {}

    inline void train() {
        _training = true;
        _children();

        for(auto& m: get_modules())
            m->_training = true;
    }

    inline void eval() {
        _training = false;
        _children();

        for(auto& m: get_modules())
            m->_training = false;
    }

    inline std::vector<Tensor<Dtype>> operator()(std::vector<Tensor<Dtype>>& inputs) {
        printf("ooooooooooo, %lu\n", get_modules().size());
        return forward(inputs);
    }

    virtual std::vector<Tensor<Dtype>> forward(std::vector<Tensor<Dtype>>& tensors) {
        std::vector<Tensor<Dtype>> inputs = tensors;
        std::vector<Tensor<Dtype>> out;

        printf("aaaaaa: %lu\n", get_modules().size());
        printf("bbbbbb: %lu\n", _sub_modules.size());

        for(int i = 0; i<get_modules().size(); ++i) {
            out = get_modules()[i]->forward(inputs);
            inputs = out;
        }

        return out;
    };

protected:
    void _children() {
        __child_modules(_sub_modules);
    }

    static std::vector<module_type>& get_modules() {
        static std::vector<module_type> _modules;
        return _modules;
    }


private:
    void __child_modules(std::vector<module_type> modules) {
        if(modules.size()==1 && modules[0]->_sub_modules.size()==0) {
            get_modules().push_back(modules[0]);
            return;
        }

        for(auto& s: _sub_modules)
            s->__child_modules({s});
    }

protected:
    bool _training;
    std::vector<module_type> _sub_modules;
    //static std::vector<module_type> _modules;
};


template<typename Dtype>
class Sequential: public Module<Dtype> {
public:
    using module_type = std::shared_ptr<Module<Dtype>>;

    Sequential(std::vector<module_type>& modules): Module<Dtype>(modules) {}

    virtual std::vector<Tensor<Dtype>> forward(std::vector<Tensor<Dtype>>& tensors) override {
        std::vector<Tensor<Dtype>> inputs = tensors;
        std::vector<Tensor<Dtype>> out;

        for(int i = 0; i<Module<Dtype>::get_modules().size(); ++i) {
            out = Module<Dtype>::get_modules()[i]->forward(inputs);
            inputs = out;
        }

        return out;
    }

};

template<typename Dtype>
class Linear: public Module<Dtype> {

public:
    Linear(int in_features, int out_features): 
        Module<Dtype>(), _in_features(in_features), _out_features(out_features) {}

    virtual std::vector<Tensor<Dtype>> forward(std::vector<Tensor<Dtype>>& tensors) override {
        printf("Linear forward: (%d, %d)\n", _in_features, _out_features);
        return {};
    }


private:
    std::vector<Tensor<Dtype>> _params;
    int _in_features;
    int _out_features;
};

#endif

