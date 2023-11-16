#ifndef UTILS_ARRAY_VIEW_H
#define UTILS_ARRAY_VIEW_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/swap.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"

namespace project_GraphFold {

template <typename T>
class Buffer {
 public:
  Buffer() = default;

  explicit Buffer(const thrust::device_vector<T>& vec)
      : data_(const_cast<T*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  explicit Buffer(const thrust::host_vector<
                  T, thrust::cuda::experimental::pinned_allocator<T>>& vec)
      : data_(const_cast<T*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  DEV_HOST Buffer(T* data, size_t size) : data_(data), size_(size) {}

  DEV_HOST_INLINE T* data() { return data_; }

  DEV_HOST_INLINE const T* data() const { return data_; }

  DEV_HOST_INLINE size_t size() const { return size_; }

  DEV_HOST_INLINE bool empty() const { return size_ == 0; }

  DEV_HOST_INLINE T& operator[](size_t i) { return data_[i]; }

  DEV_HOST_INLINE const T& operator[](size_t i) const { return data_[i]; }

  DEV_HOST_INLINE void Swap(Buffer<T>& rhs) {
    thrust::swap(data_, rhs.data_);
    thrust::swap(size_, rhs.size_);
  }

  DEV_HOST_INLINE T* begin() { return data_; }

  DEV_HOST_INLINE T* end() { return data_ + size_; }

  DEV_HOST_INLINE const T* begin() const { return data_; }

  DEV_HOST_INLINE const T* end() const { return data_ + size_; }

 private:
  T* data_{};
  size_t size_{};
};

}  // namespace project_GraphFold

#endif  // UTILS_ARRAY_VIEW_H
