#pragma once
#include "src/utils/utils.h"
// warp-wise edge-parallel: each warp takes one edge

// Update: extract graph from host and handle it on gpus.
template <typename VID, typename VLABEL>
__global__ void //__launch_bounds__(MAX_BLOCK_SIZE, 8)
tc_warp_edge(VID ne, dev::Graph<VID, VLABEL> g, AccType *total) {

  int thread_id = TID_1D;                      // global thread index
  int warp_id = thread_id / WARP_SIZE;         // global warp index
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps

  AccType count = 0;
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    auto v = g.get_src(eid);
    auto u = g.get_dst(eid);
    VID v_size = g.getOutDegree(v);
    VID u_size = g.getOutDegree(u);
    count += intersect_num(g.getNeighbor(v), v_size, g.getNeighbor(u), u_size);
  }

  atomicAdd(total, count); // atomicAdd from cmake cuda
}