#ifndef UTILS_CUDA_UTILS_H_
#define UTILS_CUDA_UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "src/utils/logging.h"
#include "src/utils/utils.h"


namespace project_GraphFold {

void SetDevice(int x) {
  cudaSetDevice(x);
  //std::cout << "Set device to " << x<<std::endl;
}

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::cout << "CUDA error " << cudaGetErrorString(err) << " at " << file
               << ":" << line<<std::endl;
  }
}

#define H_ERR(err) (HandleError(err, __FILE__, __LINE__))

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DMALLOC(pdev, bytes) H_ERR(cudaMalloc((void**) &(pdev), (bytes)))
#define TOHOST(pdev, phost, bytes) \
  H_ERR(cudaMemcpy((phost), (pdev), (bytes), D2H))
#define TODEV(pdev, phost, bytes) \
  H_ERR(cudaMemcpy((pdev), (phost), (bytes), H2D))
#define FREE(pdev) H_ERR(cudaFree((pdev)))
#define CLEAN(pdev, bytes) H_ERR(cudaMemset((pdev), 0, (bytes)))
#define WAIT() H_ERR(cudaDeviceSynchronize())

#define GET_DEV_COUNT(ndevices) \
  H_ERR(cudaGetDeviceCount((int*) &(ndevices)))

#define GET_DEV_PROP(prop, device) \
  H_ERR(cudaGetDeviceProperties(&(prop), (device)))

#define DEV_HOST __device__ __host__
#define DEV_HOST_INLINE __device__ __host__ __forceinline__
#define DEV_INLINE __device__ __forceinline__
#define MAX_BLOCK_SIZE 256
#define MAX_GRID_SIZE 768
// #define MAX_BLOCK_SIZE 256
// #define MAX_GRID_SIZE 768
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (MAX_BLOCK_SIZE / WARP_SIZE)
#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_TID_SIZE (gridDim.x * blockDim.x)

#define FULL_MASK 0xffffffff

#define ASSERT(x)                                                          \
  if (!(x)) {                                                              \
    std::cout << "Assertion failed: " << #x << " at (" << __FILE__ << ":" \
               << __LINE__ << ")"<<std::endl;                                         \
  }

// call GET_DEV_PROP first, and pass cudaDeviceProp
int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major){
    case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
    case 3: // Kepler
      cores = mp * 192;
      break;
    case 5: // Maxwell
      cores = mp * 128;
      break;
    case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
    default:
      printf("Unknown device type\n");
      break;
  }
  return cores;
}

static size_t print_device_info(bool print_all, bool verbose = false) {
  int device_num = 0;
  GET_DEV_COUNT(device_num);
  std::cout << "Total " << device_num << " GPU(s) detected.";
  size_t mem_size = 0;
  for (int device = 0; device < device_num; ++device) {
    cudaDeviceProp prop;
    SetDevice(device);
    GET_DEV_PROP(prop, device);
    if (device == 0) mem_size = prop.totalGlobalMem;
    if (!verbose) break;
    std::cout << "  Device[" << device << "]: " << prop.name;
    if (device == 0 || print_all)
    {
      std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
      std::cout << "  Warp size: " << prop.warpSize<< std::endl;
      std::cout << "  Total # SM: " << prop.multiProcessorCount<< std::endl;
      std::cout << "  Total CUDA cores: " << getSPcores(prop)<< std::endl;
      std::cout << "  Total amount of shared memory per block: " << prop.sharedMemPerBlock << " bytes"<< std::endl;
      std::cout << "  Total # registers per block: " << prop.regsPerBlock<< std::endl;
      std::cout << "  Total amount of constant memory: " << prop.totalConstMem << " bytes"<< std::endl;
      std::cout << "  Total global memory: " << float(prop.totalGlobalMem)/float(1024*1024*1024) << "GB"<< std::endl;
      std::cout << "  Memory Clock Rate: " << float(prop.memoryClockRate)/float(1024*1024) << "GHz"<< std::endl;
      std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << "bits"<< std::endl;
      std::cout << "  Peak Memory Bandwidth: " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << "GB/s"<< std::endl;
    }
  }
  return mem_size;
}

}  // namespace project_GraphFold

#if __CUDACC_VER_MAJOR__ >= 9
#define SHFL_DOWN(a, b) __shfl_down_sync(0xFFFFFFFF, a, b)
#define SHFL(a, b) __shfl_sync(0xFFFFFFFF, a, b)
#else
#define SHFL_DOWN(a, b) __shfl_down(a, b)
#define SHFL(a, b) __shfl(a, b)
#endif

//#define FREQ_PROFILE
#ifdef FREQ_PROFILE
#define PROFILE(result, size_a, size_b)\
              if(thread_lane==0) \
              (result) += (size_a) * (size_b);
#else
#define PROFILE(result, size_a, size_b)
#endif



#endif  // UTILS_CUDA_UTILS_H_
