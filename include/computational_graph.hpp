#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include "common.hpp"
#include "tensor.hpp"

namespace needle {

template<typename Dtype>
class CptGraph {
public:
    void compute_gradient_of_variables(const Tensor<Dtype>* output_tensor,
                                       const Tensor<Dtype>* out_grad) {

        //std::reverse(myVector.begin(), myVector.end());
        std::unordered_map<Tensor<Dtype>*, std::vector<Tensor<Dtype>>*> node_to_output_grads_list;

        node_to_output_grads_list[out_grad] = {out_grad};

        std::vector<Tensor<Dtype>*> topo_order = find_topo_sort({output_tensor});
        std::vector<Tensor<Dtype>*> reverse_topo_order = std::reverse(topo_order);
        // BEGEIN YOUR SOLUTION


    }

    std::vector<Tensor<Dtype>*> find_topo_sort(const std::vector<Tensor<Dtype>*> node_list) {
        std::set<Tensor<Dtype>*> visited;
        std::vector<Tensor<Dtype>*> topo_order;

        for(auto &node: node_list) {
            topo_sort_dfs(node, visited, topo_order);
        }

        return topo_order;
    }

    void topo_sort_dfs(const Tensor<Dtype> *node,
                       std::set<Tensor<Dtype>*> &visited,
                       std::vector<Tensor<Dtype>*> topo_order) {

        if(visited.find(node)) return;

        for (auto &in: node->_inputs) {
            topo_sort_dfs(in, visited, topo_order);
        }

        visited.insert(node);
        topo_order.append(node);
    }

};

}

#endif

