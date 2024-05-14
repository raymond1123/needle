#ifndef __OPTIMIZE_HPP__
#define __OPTIMIZE_HPP__

#include "nn/nn_module.cuh"

template<typename Dtype>
class Optimizer {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
    Optimizer(std::vector<cached_data_type>& params, float lr=0.01): 
        _params(params), _lr(lr) {}

    virtual void step() {}

protected:
    std::vector<cached_data_type> _params;
    float _lr;
};

template<typename Dtype>
class SGD: public Optimizer<Dtype> {
public:
    SGD(std::vector<cached_data_type>& params, float lr=0.01, 
        float weight_decay=0.0, float momentum=0.0): 
        Optimizer(params, lr), _weight_decay(weight_decay) {}

    virtual void step() override {
        for(auto& w: this->_params) {
            if(_weight_decay>0.0) {
                cached_data_type grad = w->grad + _weight_decay*w;
            } else {

            }
        }
    }

private:
    float _momentum;
    float _weight_decay;
    std::unordered_map<cached_data_type, std::vector<cached_data_type>> _update;
};

// TODO
template<typename Dtype>
class Adam: public Optimizer<Dtype> {

};

#endif

