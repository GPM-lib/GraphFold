#pragma once
// warning: from GraphMiner workload unbalanced
// vertex paralle: each warp takes one vertex
__global__ void warp_vertex(int nv, GraphGPU g, AccType *total) {
  
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  AccType count = 0;
  for (auto v = warp_id; v < nv; v += num_warps) {
    vidType *v_ptr = g.N(v);
    vidType v_size = g.getOutDegree(v);
    for (auto e = 0; e < v_size; e++) {
      auto u = v_ptr[e];
      vidType u_size = g.getOutDegree(u);
      count += intersect_num(v_ptr, v_size, g.N(u), u_size);
    }
  }

    atomicAdd(total, count);
}
