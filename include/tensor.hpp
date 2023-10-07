#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "memory.hpp"

namespace needle {
namespace NDArray{

namespace py = pybind11;

template <typename Dtype>
class Tensor {
public:
    //Tensor(py::array_t<Dtype> data,
    Tensor(const py::list &data,
           const uint32_t offset=0,
           const std::string device="cuda"): 
                _offset(offset), _device(device){

        std::cout << "offset=" << _offset << std::endl;
        std::cout << "device=" << _device << std::endl;

        _make(data);

    }

    Tensor(const Tensor &other):
        _shape(other._shape), _strides(other._strides),
        _offset(other._offset), _device(other._device),
        _handle(other._handle) {}

    Tensor(Tensor &&other):
        _shape(other._shape), _strides(other._strides),
        _offset(other._offset), _device(other._device),
        _handle(other._handle) {}

    virtual ~Tensor() {}

public:
    inline const std::vector<size_t>& shape() const {
        return _shape;
    }

    inline const std::vector<size_t>& strides() const {
        return _strides;
    }

    inline const std::string& device() const {
        return _device;
    }

    inline const size_t ndim() const {
        return _shape.size();
    }

    inline const size_t size() const {
        return _calc_size();
    }

    inline const size_t& offset() const {
        return _offset;
    }

    //Tensor compact() {
    //    if(_is_compact()) return *this;
    //}


protected:
    // Recursive function to get the shape of a nested Python list
    std::vector<size_t> _get_nested_list_shape(const py::list &lst) {
        std::vector<size_t> shape;
        size_t sublistSize = py::len(lst);

        // Add the size of the current level to the shape vector
        shape.push_back(sublistSize);

        // If the list is not empty, recursively get the shape of its first element
        if (sublistSize > 0) {
            py::object firstElement = lst[0];
            if (py::isinstance<py::list>(firstElement)) {
                // If the first element is a list, recursively get its shape
                std::vector<size_t> subShape = _get_nested_list_shape(firstElement.cast<py::list>());
                shape.insert(shape.end(), subShape.begin(), subShape.end());
            }
        }

        return shape;
    }
    // Recursive function to flatten an arbitrary nested list and convert it to float*
    void _flatten_nested_list(py::list data, Dtype *result, size_t& currentIndex) {
        for (py::handle item : data) {
            if (py::isinstance<py::list>(item)) {
                // If it's a nested list, recursively flatten it
                _flatten_nested_list(item.cast<py::list>(), result, currentIndex);
            } else {
                // If it's an int or float, convert and store in the result array
                result[currentIndex++] = py::cast<Dtype>(item);
            }
        }
    }

    inline size_t _calc_size() const {

        uint8_t size = 1;
        for(auto &s: _shape) 
            size *= s;

        return size;
    }

    void _make(const py::list &data) {
        _shape = _get_nested_list_shape(data);
        _strides = _compact_strides();

        __size = _calc_size();
        if(__size<=0) return;

        _handle = std::make_shared<Memory<Dtype>>();
        Dtype* tmp = _handle->cpu(__size);

        size_t currentIndex = 0;
        _flatten_nested_list(data, tmp, currentIndex);

        if(_device==CUDA) {
            _handle->gpu(__size);
            checkCudaErrors(cudaMemcpy(_handle->gpu(), 
                                 _handle->cpu(),
                                 __size*sizeof(Dtype),
                                 cudaMemcpyHostToDevice));
        }
    }

    inline bool _is_compact() {
        bool compact = true;
        if(_strides.size()!=_shape.size()) return false;

        auto tmp_strides = _compact_strides();
        for(int i=0; i<_strides.size(); ++i) {
            if(_strides[i] != tmp_strides[i]) {
                compact = false;
                break;
            }
        }

        size_t size = _calc_size();
        return compact && (size == _handle->cpu_size()) && (size==_handle->gpu_size());
    }

    std::vector<size_t> _compact_strides() {
        //assert(_strides.size()==_shape.size());
        std::vector<size_t> res(_shape.size());

        uint8_t stride = 1;
        for(int i=res.size()-1; i>=0; --i) {
            res[i] = stride;
            stride *= _shape[i];
        }

        return res;
    }

protected:
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
    size_t _offset;
    std::string _device; // "cpu or cuda"
    std::shared_ptr<Memory<Dtype>> _handle;

private:
    size_t __size;

};

} //namespace NDArray
} //namespace needle

#endif

