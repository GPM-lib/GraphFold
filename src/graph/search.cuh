#ifndef GRAPH_SEARCH_H
#define GRAPH_SEARCH_H
#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"

template <typename T>
__device__ int linear_search(T v, T *bin, T len, T idx, T stride) {
  auto i = idx;
  T step = 0;
  while (step < len) {
    if (bin[i] == v)
      return 1;
    else
      i += stride;
    step += 1;
  }
  return 0;
}

template <typename T>
__forceinline__ __device__ bool binary_search(T *list, T key, T size) {
  int l = 0;
  int r = size - 1;
  while (r >= l) {
    int mid = l + (r - l) / 2;
    T value = list[mid];
    if (value == key)
      return true;
    if (value < key)
      l = mid + 1;
    else
      r = mid - 1;
  }
  return false;
}

template <typename T>
__forceinline__ __device__ bool binary_search_enhanced(T *list, T key, T size) {
  int l = 0;
  int r = size - 1;
  while (r >= l) {
    int mid = l + (r - l) / 2;
    auto val = list[mid];
    if (val == key)
      return true;
    if (val < key && val >= 0)
      l = mid + 1;
    else
      r = mid - 1;
  }
  return false;
}

template <typename T>
__forceinline__ __device__ bool binary_search_2phase(T *list, T *cache, T key,
                                                     T size) {
  int p = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  int mid = 0;
  // phase 1: search in the cache
  int bottom = 0;
  int top = WARP_SIZE;
  while (top > bottom + 1) {
    mid = (top + bottom) / 2;
    T y = cache[p + mid];
    if (key == y)
      return true;
    if (key < y)
      top = mid;
    if (key > y)
      bottom = mid;
  }

  // phase 2: search in global memory
  bottom = bottom * size / WARP_SIZE;
  top = top * size / WARP_SIZE - 1;
  while (top >= bottom) {
    mid = (top + bottom) / 2;
    T y = list[mid];
    if (key == y)
      return true;
    if (key < y)
      top = mid - 1;
    else
      bottom = mid + 1;
  }
  return false;
}

template <typename T>
__forceinline__ __device__ bool binary_search_2phase_cta(T *list, T *cache,
                                                         T key, T size) {
  T y = 0; // origin VID
  int mid = 0;
  // phase 1: cache
  int bottom = 0;
  int top = MAX_BLOCK_SIZE;
  while (top > bottom + 1) {
    mid = (top + bottom) / 2;
    y = cache[mid];
    if (key == y)
      return true;
    if (key < y)
      top = mid;
    if (key > y)
      bottom = mid;
  }
  // phase 2
  bottom = bottom * size / MAX_BLOCK_SIZE;
  top = top * size / MAX_BLOCK_SIZE - 1;
  while (top >= bottom) {
    mid = (top + bottom) / 2;
    y = list[mid];
    if (key == y)
      return true;
    if (key < y)
      top = mid - 1;
    else
      bottom = mid + 1;
  }
  return false;
}

template <typename T>
__forceinline__ __device__ T binary_search_bound(T *list, T key, T size) {
  int l = 0;
  int r = size - 1;
  int mid = 0;
  T value = 0;
  while (r >= l) {
    mid = l + (r - l) / 2;
    value = list[mid];
    if (value == key)
      break;
    if (value < key)
      l = mid + 1;
    else
      r = mid - 1;
  }
  return (value >= key) ? mid : mid + 1;
}

template <typename T>
__forceinline__ __device__ bool
binary_search_2phase_map(T *list, T *cache, T key, T size, T *int_map) {
  int p = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  int mid = 0;
  // phase 1: search in the cache
  int bottom = 0;
  int top = WARP_SIZE;
  while (top > bottom + 1) {
    mid = (top + bottom) / 2;
    T y = cache[p + mid];
    if (key == y) {
      int index = mid * size / WARP_SIZE;
      int_map[index] = 1;
      return true;
    }
    if (key < y)
      top = mid;
    if (key > y)
      bottom = mid;
  }

  // phase 2: search in global memory
  bottom = bottom * size / WARP_SIZE;
  top = top * size / WARP_SIZE - 1;
  while (top >= bottom) {
    mid = (top + bottom) / 2;
    T y = list[mid];
    if (key == y) {
      int_map[mid] = 1;
      return true;
    }
    if (key < y)
      top = mid - 1;
    else
      bottom = mid + 1;
  }
  return false;
}
#endif