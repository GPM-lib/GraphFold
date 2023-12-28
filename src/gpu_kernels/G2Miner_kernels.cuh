#pragma once
#include "src/graph/search.cuh"
#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"
#include "tuning_schedules.cuh"
template <typename VID, typename VLABEL>
__global__ void P1_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                           VID max_deg, AccType *counters) {
  
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ VID list_size[WARPS_PER_BLOCK];
  AccType count = 0;
  AccType counts[6];
  for (int i = 0; i < 6; i++)
    counts[i] = 0;

  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    if (v1 == v0)
      continue;
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto cnt = difference_set(g.getNeighbor(v0), v0_size, g.getNeighbor(v1),
                              v1_size, v1, vlist);
    if (thread_lane == 0)
      list_size[warp_lane] = cnt;
    __syncwarp();
    PROFILE(counts[4], v0_size, 1);
    for (VID i = 0; i < list_size[warp_lane]; i++) {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      count += difference_num(vlist, list_size[warp_lane], g.getNeighbor(v2),
                              v2_size, v2); // 3-star
    }
    PROFILE(counts[4], list_size[warp_lane], list_size[warp_lane]);
  }
    atomicAdd(&counters[0], count);
#ifdef FREQ_PROFILE
  atomicAdd(&counters[4], counts[4]);
#endif
}
template <typename VID, typename VLABEL>
__global__ void P2_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                           VID max_deg, AccType *counters) {
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
  VID v2, v2_size;
  for (int i = 0; i < 6; i++)
    counts[i] = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][4];
  VID eid = warp_id;
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);
    auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size, &vlist[max_deg]);
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();
    PROFILE(counts[4], v0_size, 1);
    PROFILE(counts[4], v1_size, 1);
    for (VID i = 0; i < list_size[warp_lane][1]; i++) {
      VID v4 = vlist[max_deg + i];
      VID v4_size = g.getOutDegree(v4);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v4),
                           v4_size, &vlist[max_deg * 2]);
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      PROFILE(counts[4], list_size[warp_lane][0], 1);
      for (VID j = 0; j < list_size[warp_lane][2]; j++) {
        VID v2 = vlist[max_deg * 2 + j];
        VID v2_size = g.getOutDegree(v2);
        counts[0] +=
            difference_num(&vlist[max_deg * 2], list_size[warp_lane][2],
                           g.getNeighbor(v2), v2_size, v2);
      }
      PROFILE(counts[4], list_size[warp_lane][2], list_size[warp_lane][2]);
    }
  }
  atomicAdd(&counters[0], counts[0]);
#ifdef FREQ_PROFILE
  atomicAdd(&counters[4], counts[4]);
#endif
}
template <typename VID, typename VLABEL>
__global__ void P3_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                           VID max_deg, AccType *counters) {
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
  VID v2, v2_size;
  for (int i = 0; i < 6; i++)
    counts[i] = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][4];
  VID eid = warp_id;
  for (VID eid = warp_id; eid < ne; eid += num_warps)
  // for (VID eid = warp_id+78590; eid < 78590+1; eid += num_warps)
  {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);
    auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    PROFILE(counts[4], v0_size, 1);
    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v2),
                           v2_size, v2, &vlist[max_deg]);
      if (thread_lane == 0)
        list_size[warp_lane][1] = cnt;
      __syncwarp();
      PROFILE(counts[4], list_size[warp_lane][0], 1);
      for (VID j = 0; j < list_size[warp_lane][1]; j++) {
        VID v3 = vlist[max_deg + j];
        VID v3_size = g.getOutDegree(v3);
        counts[0] += difference_num(&vlist[max_deg], list_size[warp_lane][1],
                                    g.getNeighbor(v3), v3_size, v3);
      }
      PROFILE(counts[4], list_size[warp_lane][1], list_size[warp_lane][1]);
    }
  }
  atomicAdd(&counters[0], counts[0]);
#ifdef FREQ_PROFILE
  atomicAdd(&counters[4], counts[4]);
#endif
}
template <typename VID, typename VLABEL>
__global__ void P4_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                           VID max_deg, AccType *counters) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  __shared__ VID list_size[WARPS_PER_BLOCK][2];
  AccType count = 0;
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    if (v1 >= v0)
      continue;
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);
    auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist); // y0n1
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size,
                         &vlist[max_deg]); // n0y1
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (VID j = 0; j < list_size[warp_lane][0]; j++) {
      VID v2 = vlist[j];
      VID v2_size = g.getOutDegree(v2);
      count += difference_num(&vlist[max_deg], list_size[warp_lane][1],
                              g.getNeighbor(v2), v2_size); // 4-path
    }
  }
  atomicAdd(&counters[0], count);
}

template <typename VID, typename VLABEL>
__global__ void P5_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                           VID max_deg, AccType *counters) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  AccType counts[6];
  __shared__ VID v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ VID v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  VID v2, v2_size;
  for (int i = 0; i < 6; i++)
    counts[i] = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][5];
  VID eid = warp_id;
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    VID local_count = 0;
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);
    auto cnt = intersect(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();

    cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, &vlist[max_deg]);
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();

    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size, &vlist[max_deg * 2]);
    if (thread_lane == 0)
      list_size[warp_lane][2] = cnt;
    __syncwarp();

    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);

      cnt = difference_set(&vlist[max_deg * 2], list_size[warp_lane][2],
                           g.getNeighbor(v2), v2_size, &vlist[max_deg * 3]);
      if (thread_lane == 0)
        list_size[warp_lane][3] = cnt;
      __syncwarp();

      cnt = difference_set(&vlist[max_deg], list_size[warp_lane][1],
                           g.getNeighbor(v2), v2_size, &vlist[max_deg * 4]);
      if (thread_lane == 0)
        list_size[warp_lane][4] = cnt;
      __syncwarp();
      for (VID j = 0; j < list_size[warp_lane][3]; j++) {
        VID v3 = vlist[max_deg * 3 + j];
        VID v3_size = g.getOutDegree(v3);
        counts[0] +=
            difference_num(&vlist[max_deg * 4], list_size[warp_lane][4],
                           g.getNeighbor(v3), v3_size);
      }
    }
  }
  atomicAdd(&counters[0], counts[0]);
}
template <typename VID, typename VLABEL>
__global__ void P6_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                           VID max_deg, AccType *counters) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  AccType counts[6];
  __shared__ VID v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ VID v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  VID v2, v2_size;
  for (int i = 0; i < 6; i++)
    counts[i] = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][5];
  VID eid = warp_id;
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);

    auto cnt = intersect(v0_ptr, v0_size, v1_ptr, v1_size, vlist); // y0n1
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size,
                         &vlist[max_deg]); // n0y1
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();

    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v2),
                           v2_size, &vlist[max_deg * 2]); // y0n1
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      cnt = difference_set(&vlist[max_deg], list_size[warp_lane][1],
                           g.getNeighbor(v2), v2_size,
                           &vlist[max_deg * 3]); // y0n1
      if (thread_lane == 0)
        list_size[warp_lane][3] = cnt;
      __syncwarp();
      for (VID j = 0; j < list_size[warp_lane][2]; j++) {
        VID v3 = vlist[max_deg * 2 + j];
        VID v3_size = g.getOutDegree(v3);
        counts[0] +=
            difference_num(&vlist[max_deg * 3], list_size[warp_lane][3],
                           g.getNeighbor(v3), v3_size);
      }
    }
  }
  atomicAdd(&counters[0], counts[0]);
}
template <typename VID, typename VLABEL>
__global__ void P7_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                           VID max_deg, AccType *counters)

{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 3];
  AccType counts[6];
  __shared__ VID v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ VID v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  VID v2, v2_size;
  for (int i = 0; i < 6; i++)
    counts[i] = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][3];
  VID eid = warp_id;
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);

    auto cnt = intersect(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    PROFILE(counts[4], v0_size, 1);
    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);

      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v2),
                           v2_size, v2, &vlist[max_deg]); // y0n1
      if (thread_lane == 0)
        list_size[warp_lane][1] = cnt;
      __syncwarp();
      PROFILE(counts[4], list_size[warp_lane][0], 1);
      for (VID j = 0; j < list_size[warp_lane][1]; j++) {
        VID v3 = vlist[max_deg * 1 + j];
        VID v3_size = g.getOutDegree(v3);
        counts[0] +=
            difference_num(&vlist[max_deg * 1], list_size[warp_lane][1],
                           g.getNeighbor(v3), v3_size, v3);
      }
      PROFILE(counts[4], list_size[warp_lane][1], list_size[warp_lane][1]);
    }
  }
  atomicAdd(&counters[0], counts[0]);
#ifdef FREQ_PROFILE
  atomicAdd(&counters[4], counts[4]);
#endif
}
template <typename VID, typename VLABEL>
__global__ void P8_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                           VID max_deg, AccType *counters) {
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
  VID v2, v2_size;
  for (int i = 0; i < 6; i++)
    counts[i] = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][4];
  VID eid = warp_id;
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);

    auto cnt = intersect(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size, &vlist[max_deg]);
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();
    PROFILE(counts[4], v0_size, 1);
    PROFILE(counts[4], v1_size, 1);
    for (VID i = 0; i < list_size[warp_lane][1]; i++) {
      VID v4 = vlist[max_deg + i];
      VID v4_size = g.getOutDegree(v4);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v4),
                           v4_size, &vlist[max_deg * 2]); // y0n1
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      PROFILE(counts[4], list_size[warp_lane][0], 1);
      for (VID j = 0; j < list_size[warp_lane][2]; j++) {
        VID v3 = vlist[max_deg * 2 + j];
        VID v3_size = g.getOutDegree(v3);
        counts[0] += intersect_num(&vlist[max_deg * 2], list_size[warp_lane][2],
                                   g.getNeighbor(v3), v3_size, v3);
      }
      PROFILE(counts[4], list_size[warp_lane][2], list_size[warp_lane][2]);
    }
  }
  atomicAdd(&counters[0], counts[0]);
#ifdef FREQ_PROFILE
  atomicAdd(&counters[4], counts[4]);
#endif
}

template <typename VID, typename VLABEL>
__global__ void clique4_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                                size_t max_deg, AccType *total) {
  
  int thread_id = TID_1D;
  int warp_id = thread_id / WARP_SIZE; // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg)];
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK];
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto count = intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1),
                           v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane] = count;
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane]; i++) {
      VID u = vlist[i];
      VID u_size = g.getOutDegree(u);
      VID v_size = list_size[warp_lane];
      counter += intersect_num(vlist, v_size, g.getNeighbor(u), u_size);
    }
  }
atomicAdd(total, counter);
}

template <typename VID, typename VLABEL>
__global__ void clique5_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                                VID max_deg, AccType *total) {
  
  int thread_id = TID_1D;
  int warp_id = thread_id / WARP_SIZE; // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][2];
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto v0_size = g.getOutDegree(v0); // ori: get_degree return VertexID type
    auto v1_size = g.getOutDegree(v1); // ori: get_degree return VertexID type
    auto count1 = intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1),
                            v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = count1;
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      auto v2 = vlist[i];
      auto v2_size = g.getOutDegree(v2); // ori: get_degree return VertexID type
      auto w1_size = list_size[warp_lane][0];
      auto count2 = intersect(vlist, w1_size, g.getNeighbor(v2), v2_size,
                              vlist + max_deg);
      if (thread_lane == 0)
        list_size[warp_lane][1] = count2;
      __syncwarp();
      for (VID j = 0; j < list_size[warp_lane][1]; j++) {
        auto v3 = vlist[max_deg + j];
        auto v3_size =
            g.getOutDegree(v3); // ori: get_degree return VertexID type
        auto w2_size = list_size[warp_lane][1];
        counter +=
            intersect_num(vlist + max_deg, w2_size, g.getNeighbor(v3), v3_size);
      }
    }
  }
atomicAdd(total, counter);
}

template <typename VID, typename VLABEL>
__global__ void clique6_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                                VID max_deg, AccType *total) {
  
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE; // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps

  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 3];
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][3];
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto count1 = intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1),
                            v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = count1;
    __syncwarp();
    for (VID idx1 = 0; idx1 < list_size[warp_lane][0]; idx1++) {
      VID v2 = vlist[idx1];
      VID v2_size = g.getOutDegree(v2);
      VID w1_size = list_size[warp_lane][0];
      auto count2 = intersect(vlist, w1_size, g.getNeighbor(v2), v2_size,
                              vlist + max_deg);
      if (thread_lane == 0)
        list_size[warp_lane][1] = count2;
      __syncwarp();
      for (VID idx2 = 0; idx2 < list_size[warp_lane][1]; idx2++) {
        VID v3 = vlist[max_deg + idx2];
        VID v3_size = g.getOutDegree(v3);
        VID w2_size = list_size[warp_lane][1];
        auto count3 = intersect(vlist + max_deg, w2_size, g.getNeighbor(v3),
                                v3_size, vlist + max_deg * 2);
        if (thread_lane == 0)
          list_size[warp_lane][2] = count3;
        __syncwarp();
        for (VID idx3 = 0; idx3 < list_size[warp_lane][2]; idx3++) {
          VID v4 = vlist[max_deg * 2 + idx3];
          VID v4_size = g.getOutDegree(v4);
          VID w3_size = list_size[warp_lane][2];
          counter += intersect_num(vlist + max_deg * 2, w3_size,
                                   g.getNeighbor(v4), v4_size);
        }
      }
    }
  }
atomicAdd(total, counter);
}

template <typename VID, typename VLABEL>
__global__ void clique7_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists,
                                VID max_deg, AccType *total) {
  
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE; // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps

  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 4];
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][4];
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto count1 = intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1),
                            v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = count1;
    __syncwarp();
    for (VID idx1 = 0; idx1 < list_size[warp_lane][0]; idx1++) {
      VID v2 = vlist[idx1];
      VID v2_size = g.getOutDegree(v2);
      VID w1_size = list_size[warp_lane][0];
      auto count2 = intersect(vlist, w1_size, g.getNeighbor(v2), v2_size,
                              vlist + max_deg);
      if (thread_lane == 0)
        list_size[warp_lane][1] = count2;
      __syncwarp();
      for (VID idx2 = 0; idx2 < list_size[warp_lane][1]; idx2++) {
        VID v3 = vlist[max_deg + idx2];
        VID v3_size = g.getOutDegree(v3);
        VID w2_size = list_size[warp_lane][1];
        auto count3 = intersect(vlist + max_deg, w2_size, g.getNeighbor(v3),
                                v3_size, vlist + max_deg * 2);
        if (thread_lane == 0)
          list_size[warp_lane][2] = count3;
        __syncwarp();
        for (VID idx3 = 0; idx3 < list_size[warp_lane][2]; idx3++) {
          VID v4 = vlist[max_deg * 2 + idx3];
          VID v4_size = g.getOutDegree(v4);
          VID w3_size = list_size[warp_lane][2];
          auto count4 =
              intersect(vlist + max_deg * 2, w3_size, g.getNeighbor(v4),
                        v4_size, vlist + max_deg * 3);
          if (thread_lane == 0)
            list_size[warp_lane][3] = count4;
          __syncwarp();
          for (VID idx4 = 0; idx4 < list_size[warp_lane][3]; idx4++) {
            VID v5 = vlist[max_deg * 3 + idx4];
            VID v5_size = g.getOutDegree(v5);
            VID w4_size = list_size[warp_lane][3];
            counter += intersect_num(vlist + max_deg * 3, w4_size,
                                     g.getNeighbor(v5), v5_size);
          }
        }
      }
    }
  }
atomicAdd(total, counter);
}

template <typename VID, typename VLABEL>
__global__ void tc_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, AccType *total) {
  
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps

  AccType count = 0;
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    auto v = g.get_src(eid);
    auto u = g.get_dst(eid);
    VID v_size = g.getOutDegree(v);
    VID u_size = g.getOutDegree(u);
    count += intersect_num(g.getNeighbor(v), v_size, g.getNeighbor(u), u_size);
  }
 
    atomicAdd(total, count);
}

template <typename VID, typename VLABEL>
__global__ void motif_G2Miner(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType* counters) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;            // total number of active warps
  VID* vlist = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  VID counts[6];
  __shared__ VID v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ VID v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  VID v2, v2_size;
  for (int i = 0; i < 6; i++) counts[i] = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][3];
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    if (thread_lane == 0) {
      v0[warp_lane] = g.get_src(eid);
      v1[warp_lane] = g.get_dst(eid);
    }
    __syncwarp();
    if (v1[warp_lane] == v0[warp_lane]) continue;
    if (thread_lane == 0) {
      v0_size[warp_lane] = g.getOutDegree(v0[warp_lane]);
      v1_size[warp_lane] = g.getOutDegree(v1[warp_lane]);
    }
    __syncwarp();
    auto v0_ptr = g.getNeighbor(v0[warp_lane]);
    auto v1_ptr = g.getNeighbor(v1[warp_lane]);

    // finding 3-star
    auto cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], v1[warp_lane], vlist);
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      v2 = vlist[i];
      v2_size = g.getOutDegree(v2);
      counts[0] += difference_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v2); // 3-star
    }

    if (v1[warp_lane] > v0[warp_lane]) continue;

    // finding diamond and tailed_triangle
    cnt = intersect(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist); // y0y1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (VID j = 0; j < list_size[warp_lane][0]; j++) {
      v2 = vlist[j];
      v2_size = g.getOutDegree(v2);
      counts[4] += difference_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v2); // diamond

      cnt = difference_set(g.getNeighbor(v2), v2_size, v0_ptr, v0_size[warp_lane], &vlist[max_deg]); // n0y2
      if (thread_lane == 0) list_size[warp_lane][1] = cnt;
      __syncwarp();
      counts[2] += difference_num(&vlist[max_deg], list_size[warp_lane][1], v1_ptr, v1_size[warp_lane]); // n0n1y2: tailed_triangle
    }

    // finding 4-clique
    cnt = intersect(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], v1[warp_lane], vlist); // y0f0y1f1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      v2 = vlist[i];
      v2_size = g.getOutDegree(v2);
      counts[5] += intersect_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v2); // 4-cycle
    }

    // finding 4-path
    cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist); // y0n1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane], &vlist[max_deg]); // n0y1
    if (thread_lane == 0) list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (VID j = 0; j < list_size[warp_lane][0]; j++) {
      v2 = vlist[j];
      v2_size = g.getOutDegree(v2);
      counts[1] += difference_num(&vlist[max_deg], list_size[warp_lane][1], g.getNeighbor(v2), v2_size); // 4-path
    }

    // finding 4-cycle
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane], v0[warp_lane], vlist); // n0f0y1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], v1[warp_lane], &vlist[max_deg]); // y0f0n1f1
    if (thread_lane == 0) list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (VID j = 0; j < list_size[warp_lane][1]; j++) {
      v2 = vlist[max_deg+j];
      v2_size = g.getOutDegree(v2);
      counts[3] += intersect_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v0[warp_lane]); // 4-cycle
    }
  }
  for (int i = 0; i < 6; i++)
    atomicAdd(&counters[i], counts[i]);
}
