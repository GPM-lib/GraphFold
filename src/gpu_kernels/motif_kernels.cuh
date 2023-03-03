#pragma once
#include "tuning_schedules.cuh"

__global__ void sub(AccType *accumulators) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  accumulators[0] = accumulators[1] - accumulators[2];
  accumulators[1] = 0;
  accumulators[2] = 0;
}

__global__ void restore(AccType *accumulators) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  accumulators[0] = accumulators[1];
  accumulators[6] = accumulators[2];
  accumulators[1] = 0;
  accumulators[2] = 0;
}

template <typename VID, typename VLABEL>
__global__ void motif4_warp_edge_bsopt_part_woPATH4_fusion(
    VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg,
    AccType *counters, AccType *INDEX, int *int_maps)

{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 4];
  AccType counts[6];
  __shared__ VID v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ VID v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];

  VID *int_map = &int_maps[int64_t(warp_id) * int64_t(max_deg)];

  VID v2, v2_size;
  for (int i = 0; i < 6; i++)
    counts[i] = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][4];

  VID eid = warp_id;
//#define NE filter_cnt
#define NE ne
  while (eid < NE) {
    if (thread_lane == 0) {
      v0[warp_lane] = g.get_src(eid);
      v1[warp_lane] = g.get_dst(eid);
    }
    __syncwarp();
    if (v1[warp_lane] >= v0[warp_lane]) {
      NEXT_WORK_CATCH(eid, INDEX, num_warps);
      continue;
    }

    if (thread_lane == 0) {
      v0_size[warp_lane] = g.getOutDegree(v0[warp_lane]);
      v1_size[warp_lane] = g.getOutDegree(v1[warp_lane]);
    }
    __syncwarp();
    auto v0_ptr = g.getNeighbor(v0[warp_lane]);
    auto v1_ptr = g.getNeighbor(v1[warp_lane]);

    // calculate N(v0)-N(v1) -> vlist
    // N(v0)∩N(v1) -> vlist+max_deg
    // N(v1)-N(v1) -> vlist+max_deg*2
    auto int01_cnt =
        fuse_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist,
                 &vlist[max_deg], &vlist[max_deg * 2], int_map); // y0y1

    int cnt = 0;
    if (thread_lane == 0)
      list_size[warp_lane][0] = int01_cnt;
    __syncwarp();
    auto dif01_set = vlist;
    auto int01_set = &vlist[max_deg];
    auto dif10_set = &vlist[max_deg * 2];
    auto tmp_set = &vlist[max_deg * 3];
    auto bound_set = &vlist[max_deg * 5];

    for (VID j = 0; j < list_size[warp_lane][0]; j++) {
      v2 = int01_set[j];
      v2_size = g.getOutDegree(v2);
      // counting diamond(counts[4]) and 4-clique(counts[5]),
      // v2,v3=N(v0)∩N(v1)
      // if v2-v3 not connect, then count diamond
      // if v2-v3 connect, then count 4-clique
      for (auto i = thread_lane; i < list_size[warp_lane][0]; i += WARP_SIZE) {
        auto key = int01_set[i];
        int is_smaller = key < v2 ? 1 : 0;
        // Notice: use direct bs can be fast than shared bs here
        int flag = !binary_search(g.getNeighbor(v2), key, v2_size);
        counts[4] += (flag)&is_smaller;
        counts[5] += ((1 - flag) & is_smaller & (v2 < v1[warp_lane]) &
                      key < v1[warp_lane]);
      }

      // counting tailed(counts[2])
      // v2=N(v0)∩N(v1)
      // v3=N(v2)-N(v0)
      // if v1-v3 not connect, then count tailed
      cnt = difference_set(g.getNeighbor(v2), v2_size, v0_ptr,
                           v0_size[warp_lane], tmp_set); // n0y2
      if (thread_lane == 0)
        list_size[warp_lane][1] = cnt;
      __syncwarp();
      for (auto i = thread_lane; i < list_size[warp_lane][1]; i += WARP_SIZE) {
        auto key = tmp_set[i];
        // Notice: use direct bs can be fast than shared bs here
        if (!binary_search(v1_ptr, key, v1_size[warp_lane]))
          counts[2] += 1;
      }
    }
    // counting 4-path(counts[1]) and 4-cycle(counts[3])
    // v2=N(v0)-N(v1)
    // v3=N(v1)-N(v0)
    // if v2-v3 not connect, then count 4-path
    // if v2-v3 connect, then count 4-cycle
    if (thread_lane == 0) {
      list_size[warp_lane][0] = v0_size[warp_lane] - int01_cnt;
      list_size[warp_lane][1] = v1_size[warp_lane] - int01_cnt;
    }
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane],
                         v0[warp_lane], vlist); // n0f0y1
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane],
                         v1[warp_lane], &vlist[max_deg]); // y0f0n1f1
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (VID j = 0; j < list_size[warp_lane][1]; j++) {
      v2 = vlist[max_deg + j];
      v2_size = g.getOutDegree(v2);
      counts[3] +=
          intersect_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2),
                        v2_size, v0[warp_lane]); // 4-cycle
    }

    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }

  for (int i = 0; i < 6; i++)
    atomicAdd(&counters[i], counts[i]);
}
