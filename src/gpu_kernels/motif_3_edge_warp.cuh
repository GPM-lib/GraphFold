#pragma once
// edge parallel: each warp takes one edge
#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"

template <typename VID, typename VLABEL>
__device__ void motif_3_warp_edge(VID ne, dev::Graph<VID, VLABEL> g,
                                  size_t max_deg, AccType *total) {
  
  int thread_id = TID_1D;                      // global thread id
  int warp_id = thread_id / WARP_SIZE;         // global warp index
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  AccType tri_count = 0;
  AccType wed_count = 0;
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    if (v1 == v0)
      continue;
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    wed_count += difference_num(g.getNeighbor(v0), v0_size, g.getNeighbor(v1),
                                v1_size, v1);
    if (v1 >= v0)
      continue;
    tri_count += intersect_num(g.getNeighbor(v0), v0_size, g.getNeighbor(v1),
                               v1_size, v1);
  }

    atomicAdd(&total[0], tri_count);
    atomicAdd(&total[1], wed_count);
}
