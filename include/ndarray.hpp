#ifndef __NDARRAY_HPP__
#define __NDARRAY_HPP__

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "memory.hpp"
#include "ops.cuh"

namespace needle {
//namespace NDArray{

namespace py = pybind11;

template <typename Dtype>
class NDArray{
public:
    NDArray(const py::list &data,
           const uint32_t offset=0,
           const std::string device="cuda"): 
                _offset(offset), _device(device){

        _make_from_list(data);

    }

    NDArray(const std::shared_ptr<Memory<Dtype>> data,
           const std::vector<size_t> shape,
           const std::vector<size_t> strides,
           const uint32_t offset=0,
           const std::string device="cuda"): 
                _shape(shape), _strides(strides),
                _offset(offset), _device(device) {

        __size = _calc_size();
        _make_from_tensor(data);

    }

    NDArray(const NDArray &other):
        _shape(other._shape), _strides(other._strides),
        _offset(other._offset), _device(other._device),
        __size(other.__size), _handle(other._handle) {
            std::cout << "in normal constructor" << std::endl;
        }

    NDArray(NDArray &&other):
        _shape(other._shape), _strides(other._strides),
        _offset(other._offset), _device(other._device),
        __size(other.__size), _handle(other._handle) {
            std::cout << "in move constructor" << std::endl;
        }

    virtual ~NDArray() {}

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

    std::string print() {
        std::ostringstream stream;
        __gpu2cpu(_handle, _handle);

        Dtype *data = _handle->cpu();

        stream << "data: [";
        for(size_t i=0; i<__size; ++i) {
            stream << data[i] << ", ";
        }
        stream << "]";

        return stream.str();
    }

    /* operations */
    NDArray operator+(const NDArray &other) const {
        NDArray res(other._handle, other._shape, 
                   other._strides, other._offset, 
                   other._device);

        EwiseAdd<Dtype> edd;
        if(_device==CPU)
            edd.compute_cpu(reinterpret_cast<Dtype*>(_handle->cpu()), 
                       reinterpret_cast<Dtype*>(other._handle->cpu()),
                       reinterpret_cast<Dtype*>(res._handle->cpu()),
                       __size);

        if(_device==CUDA) {
            edd.compute_gpu(reinterpret_cast<Dtype*>(_handle->gpu()), 
                       reinterpret_cast<Dtype*>(other._handle->gpu()),
                       reinterpret_cast<Dtype*>(res._handle->gpu()),
                       __size);
        }

        return res;
    }

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

    void _make_from_tensor(const std::shared_ptr<Memory<Dtype>> data) {
        _handle = std::make_shared<Memory<Dtype>>();

        if(_device==CPU) {
            _handle->cpu(__size);
            std::memcpy(_handle->cpu(), data->cpu(), __size*sizeof(Dtype));
        }

        if(_device==CUDA) {
            _handle->gpu(__size);
            __gpu2gpu(data, _handle);
        }
    }

    void _make_from_list(const py::list &data) {
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
            __cpu2gpu(_handle, _handle);
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

private:
    void __cpu2gpu(std::shared_ptr<Memory<Dtype>> src, 
                   std::shared_ptr<Memory<Dtype>> dst) {
        if(!dst->own_gpu())
            dst->gpu(src->cpu_size());

        __size = src->cpu_size();
        if(src->own_cpu() && dst->own_gpu()) {
            checkCudaErrors(cudaMemcpy(dst->gpu(),
                                       src->cpu(),
                                       __size*sizeof(Dtype),
                                       cudaMemcpyHostToDevice));
        }
    }

    void __gpu2cpu(std::shared_ptr<Memory<Dtype>> src, 
                   std::shared_ptr<Memory<Dtype>> dst) {
        if(!dst->own_cpu()) 
            dst->cpu(src->gpu_size());
        __size = src->gpu_size();

        if(src->own_gpu() && dst->own_cpu()) {
            checkCudaErrors(cudaMemcpy(dst->cpu(),
                                       src->gpu(),
                                       __size*sizeof(Dtype),
                                       cudaMemcpyDeviceToHost));
        }
    }

    void __gpu2gpu(std::shared_ptr<Memory<Dtype>> src, 
                   std::shared_ptr<Memory<Dtype>> dst) {

        if(!dst->own_gpu()) 
            dst->gpu(src->gpu_size());
        __size = src->gpu_size();

        if(src->own_gpu() && dst->own_gpu()) {
            checkCudaErrors(cudaMemcpy(dst->gpu(),
                                       src->gpu(),
                                       __size*sizeof(Dtype),
                                       cudaMemcpyDeviceToDevice));
        }
    }

protected:
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
    size_t _offset;
    std::string _device; // "cpu or cuda"
    std::shared_ptr<Memory<Dtype>> _handle;

private:
    mutable size_t __size;

};

//} //namespace NDArray
} //namespace needle

#endif

