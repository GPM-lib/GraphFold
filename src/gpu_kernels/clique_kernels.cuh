// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking
// on-the-fly
#pragma once
#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"

#include "src/utils/bitsets.h"
#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"

#include "src/gpu_kernels/tuning_schedules.cuh"

template <typename VID, typename VLABEL>
__global__ void clique5_graphfold(VID ne, dev::Graph<VID, VLABEL> g, VID *vmaps,
                                  MultiBitsets<> adj_lists, VID max_deg,
                                  AccType *total, AccType *INDEX) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;         // global warp index
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  VID *vmap = &vmaps[int64_t(warp_id) * int64_t(max_deg)];
  size_t offset = warp_id * max_deg * ((max_deg - 1) / 32 + 1);
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK];
  __shared__ uint32_t bitmap32[WARPS_PER_BLOCK][32];
  __shared__ uint32_t bitmap64[WARPS_PER_BLOCK][128];
  __shared__ uint64_t bitmap__[WARPS_PER_BLOCK][64];
  for (VID eid = warp_id; eid < ne;) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    VID local_count =
        intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1), v1_size, vmap);
    if (thread_lane == 0)
      list_size[warp_lane] = local_count;
    __syncwarp();
    VID count = list_size[warp_lane];

#ifdef BITMAP_OPT
    if (count <= 32) {
      for (VID i = 0; i < count; i++) {
        auto search = g.getNeighbor(vmap[i]);
        VID search_size = g.getOutDegree(vmap[i]);
        if (thread_lane == 0)
          bitmap32[warp_lane][i] = 0;
        __syncwarp();
        for (auto j = thread_lane; j < count; j += WARP_SIZE) {
          unsigned active = __activemask();
          bool flag = (j != i) && binary_search(search, vmap[j], search_size);
          __syncwarp(active);
          unsigned mask = __ballot_sync(active, flag);
          if (thread_lane == 0)
            bitmap32[warp_lane][i] = mask;
        }
      }
      for (VID i = thread_lane; i < count; i += 32) {
        VID v2 = vmap[i];
        auto to_check = bitmap32[warp_lane][i];
        for (int j = 0; j < 32; j++) {
          if (to_check & (1 << j)) {
            counter += __popc(bitmap32[warp_lane][j] & to_check);
          }
        }
      }
    }

    else if (count <= 64) {
      for (VID i = 0; i < count; i++) {
        auto search = g.getNeighbor(vmap[i]);
        VID search_size = g.getOutDegree(vmap[i]);
        if (thread_lane == 0) {
          bitmap64[warp_lane][i * 2 + 0] = 0;
          bitmap64[warp_lane][i * 2 + 1] = 0;
        }
        __syncwarp();

        if (thread_lane < count) {
          unsigned active = __activemask();
          auto j = thread_lane;
          bool flag = (j != i) && binary_search(search, vmap[j], search_size);
          __syncwarp(active);
          unsigned mask = __ballot_sync(active, flag);
          if (thread_lane == 0)
            bitmap64[warp_lane][i * 2 + 0] = mask;
        }
        __syncwarp();
        if (thread_lane + 32 < count) {
          unsigned active = __activemask();
          auto j = thread_lane + 32;
          bool flag = (j != i) && binary_search(search, vmap[j], search_size);
          __syncwarp(active);
          unsigned mask = __ballot_sync(active, flag);
          if (thread_lane == 0)
            bitmap64[warp_lane][i * 2 + 1] = mask;
        }
        __syncwarp();
      }
      __syncwarp();

      if (thread_lane < count) {

        VID v2 = vmap[thread_lane];
        auto to_check0 = bitmap64[warp_lane][thread_lane * 2 + 0];
        auto to_check1 = bitmap64[warp_lane][thread_lane * 2 + 1];
        for (int j = 0; j < 32; j++) {
          if (to_check0 & (1 << j)) {
            counter += __popc(bitmap64[warp_lane][j * 2 + 0] & to_check0);
            counter += __popc(bitmap64[warp_lane][j * 2 + 1] & to_check1);
          }
          if (to_check1 & (1 << j)) {
            counter +=
                __popc(bitmap64[warp_lane][(j + 32) * 2 + 0] & to_check0);
            counter +=
                __popc(bitmap64[warp_lane][(j + 32) * 2 + 1] & to_check1);
          }
        }
      }
      __syncwarp();
      if (thread_lane + 32 < count) {
        VID v2 = vmap[thread_lane + 32];
        auto to_check0 = bitmap64[warp_lane][(thread_lane + 32) * 2 + 0];
        auto to_check1 = bitmap64[warp_lane][(thread_lane + 32) * 2 + 1];
        for (int j = 0; j < 32; j++) {
          if (to_check0 & (1 << j)) {
            counter += __popc(bitmap64[warp_lane][j * 2 + 0] & to_check0);
            counter += __popc(bitmap64[warp_lane][j * 2 + 1] & to_check1);
          }
          if (to_check1 & (1 << j)) {
            counter +=
                __popc(bitmap64[warp_lane][(j + 32) * 2 + 0] & to_check0);
            counter +=
                __popc(bitmap64[warp_lane][(j + 32) * 2 + 1] & to_check1);
          }
        }
      }
    } else
#endif
    {
      for (VID i = 0; i < count; i++) {
        auto search = g.getNeighbor(vmap[i]);
        VID search_size = g.getOutDegree(vmap[i]);
        for (auto j = thread_lane; j < count; j += WARP_SIZE) {
          unsigned active = __activemask();
          bool flag = (j != i) && binary_search(search, vmap[j], search_size);
          __syncwarp(active);
          adj_lists.warp_set(offset, i, j, flag);
        }
      }
      // auto count = construct_induced_graph(g, v0, v1, adj_lists, vlists);
      __syncwarp();
      auto nc = (count - 1) / 32 + 1;
      for (VID v2 = 0; v2 < count; v2++) {
        for (VID v3 = 0; v3 < count; v3++) {
          if (adj_lists.get(offset, v2, v3)) {
            auto count1 = adj_lists.intersect_num(offset, nc, v2, v3);
            // if (thread_lane == 0) {
            // if (count1 > 0) printf("v0=%d, v1=%d, v2=%d, v3=%d, count=%d\n",
            // v0, v1, vmap[v2], vmap[v3], count1);
            counter += count1;
            //}
          }
        }
      }
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  atomicAdd(total, counter);
}

template <typename VID, typename VLABEL>
__global__ void clique6_graphfold(VID ne, dev::Graph<VID, VLABEL> g, VID *vmaps,
                                  MultiBitsets<> adj_lists, VID max_deg,
                                  AccType *total, AccType *INDEX) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;         // global warp index
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps

  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  VID *vmap = &vmaps[int64_t(warp_id) * int64_t(max_deg) * 2];
  size_t offset = warp_id * max_deg * ((max_deg - 1) / 32 + 1);
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][2];

  __shared__ uint32_t bitmap32[WARPS_PER_BLOCK][32];

  __shared__ uint32_t bitmap64[WARPS_PER_BLOCK][128];
  for (VID eid = warp_id; eid < ne;) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto count =
        intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1), v1_size, vmap);
    if (thread_lane == 0)
      list_size[warp_lane][0] = count;
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      VID u = vmap[i];
      VID u_size = g.getOutDegree(u);
      VID v_size = list_size[warp_lane][0];
      VID local_count =
          intersect(vmap, v_size, g.getNeighbor(u), u_size, &vmap[max_deg]);
      if (thread_lane == 0)
        list_size[warp_lane][1] = local_count;
      __syncwarp();
      VID count = list_size[warp_lane][1];

#ifdef BITMAP_OPT

      // sm store 64 elements!!!!!!!
      if (count <= 32) {
        for (VID i = 0; i < count; i++) {
          auto search = g.getNeighbor(vmap[max_deg + i]);
          VID search_size = g.getOutDegree(vmap[max_deg + i]);
          if (thread_lane == 0)
            bitmap32[warp_lane][i] = 0;
          __syncwarp();
          for (auto j = thread_lane; j < count; j += WARP_SIZE) {
            unsigned active = __activemask();
            bool flag = (j != i) &&
                        binary_search(search, vmap[max_deg + j], search_size);
            __syncwarp(active);
            unsigned mask = __ballot_sync(active, flag);
            if (thread_lane == 0)
              bitmap32[warp_lane][i] = mask;
          }
        }
        for (VID i = thread_lane; i < count; i += 32) {
          VID v2 = vmap[max_deg + i];
          auto to_check = bitmap32[warp_lane][i];
          for (int j = 0; j < 32; j++) {
            if (to_check & (1 << j)) {
              counter += __popc(bitmap32[warp_lane][j] & to_check);
            }
          }
        }
      }

      else if (count <= 64) {
        for (VID i = 0; i < count; i++) {
          auto search = g.getNeighbor(vmap[max_deg + i]);
          VID search_size = g.getOutDegree(vmap[max_deg + i]);
          if (thread_lane == 0) {
            bitmap64[warp_lane][i * 2 + 0] = 0;
            bitmap64[warp_lane][i * 2 + 1] = 0;
          }
          __syncwarp();

          if (thread_lane < count) {
            unsigned active = __activemask();
            auto j = thread_lane;
            bool flag = (j != i) &&
                        binary_search(search, vmap[max_deg + j], search_size);
            __syncwarp(active);
            unsigned mask = __ballot_sync(active, flag);
            if (thread_lane == 0)
              bitmap64[warp_lane][i * 2 + 0] = mask;
          }
          __syncwarp();
          if (thread_lane + 32 < count) {
            unsigned active = __activemask();
            auto j = thread_lane + 32;
            bool flag = (j != i) &&
                        binary_search(search, vmap[max_deg + j], search_size);
            __syncwarp(active);
            unsigned mask = __ballot_sync(active, flag);
            if (thread_lane == 0)
              bitmap64[warp_lane][i * 2 + 1] = mask;
          }
          __syncwarp();
        }
        __syncwarp();

        if (thread_lane < count) {

          VID v2 = vmap[max_deg + thread_lane];
          auto to_check0 = bitmap64[warp_lane][thread_lane * 2 + 0];
          auto to_check1 = bitmap64[warp_lane][thread_lane * 2 + 1];
          for (int j = 0; j < 32; j++) {
            if (to_check0 & (1 << j)) {
              counter += __popc(bitmap64[warp_lane][j * 2 + 0] & to_check0);
              counter += __popc(bitmap64[warp_lane][j * 2 + 1] & to_check1);
            }
            if (to_check1 & (1 << j)) {
              counter +=
                  __popc(bitmap64[warp_lane][(j + 32) * 2 + 0] & to_check0);
              counter +=
                  __popc(bitmap64[warp_lane][(j + 32) * 2 + 1] & to_check1);
            }
          }
        }
        __syncwarp();
        if (thread_lane + 32 < count) {
          VID v2 = vmap[max_deg + thread_lane + 32];
          auto to_check0 = bitmap64[warp_lane][(thread_lane + 32) * 2 + 0];
          auto to_check1 = bitmap64[warp_lane][(thread_lane + 32) * 2 + 1];
          for (int j = 0; j < 32; j++) {
            if (to_check0 & (1 << j)) {
              counter += __popc(bitmap64[warp_lane][j * 2 + 0] & to_check0);
              counter += __popc(bitmap64[warp_lane][j * 2 + 1] & to_check1);
            }
            if (to_check1 & (1 << j)) {
              counter +=
                  __popc(bitmap64[warp_lane][(j + 32) * 2 + 0] & to_check0);
              counter +=
                  __popc(bitmap64[warp_lane][(j + 32) * 2 + 1] & to_check1);
            }
          }
        }
      } else

#endif
      // continue;
      {
        for (VID i = 0; i < count; i++) {
          auto search = g.getNeighbor(vmap[max_deg + i]);
          VID search_size = g.getOutDegree(vmap[max_deg + i]);
          for (auto j = thread_lane; j < count; j += WARP_SIZE) {
            unsigned active = __activemask();
            bool flag = (j != i) &&
                        binary_search(search, vmap[max_deg + j], search_size);
            __syncwarp(active);
            adj_lists.warp_set(offset, i, j, flag);
          }
        }
        // auto count = construct_induced_graph(g, v0, v1, adj_lists, vlists);
        __syncwarp();
        auto nc = (count - 1) / 32 + 1;
        for (VID v2 = 0; v2 < count; v2++) {
          for (VID v3 = 0; v3 < count; v3++) {
            if (adj_lists.get(offset, v2, v3)) {
              auto count1 = adj_lists.intersect_num(offset, nc, v2, v3);
              // if (thread_lane == 0) {
              // if (count1 > 0) printf("v0=%d, v1=%d, v2=%d, v3=%d,
              // count=%d\n", v0, v1, vmap[v2], vmap[v3], count1);
              counter += count1;
              //}
            }
          }
        }
      }
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  atomicAdd(total, counter);
}

template <typename VID, typename VLABEL>
__global__ void clique7_graphfold(VID ne, dev::Graph<VID, VLABEL> g, VID *vmaps,
                                  MultiBitsets<> adj_lists, VID max_deg,
                                  AccType *total, AccType *INDEX) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;         // global warp index
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps

  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  VID *vmap = &vmaps[int64_t(warp_id) * int64_t(max_deg) * 3];
  size_t offset = warp_id * max_deg * ((max_deg - 1) / 32 + 1);
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][3];

  __shared__ uint32_t bitmap32[WARPS_PER_BLOCK][32];

  __shared__ uint32_t bitmap64[WARPS_PER_BLOCK][128];
  for (VID eid = warp_id; eid < ne;) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto count =
        intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1), v1_size, vmap);
    if (thread_lane == 0)
      list_size[warp_lane][0] = count;
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      VID u = vmap[i];
      VID u_size = g.getOutDegree(u);
      VID v_size = list_size[warp_lane][0];

      auto count =
          intersect(vmap, v_size, g.getNeighbor(u), u_size, &vmap[max_deg * 2]);
      if (thread_lane == 0)
        list_size[warp_lane][2] = count;
      __syncwarp();
      for (VID j = 0; j < list_size[warp_lane][2]; j++) {
        VID kkk = vmap[max_deg * 2 + j];
        VID kkk_size = g.getOutDegree(kkk);
        VID xxx_size = list_size[warp_lane][2];

        VID local_count =
            intersect(&vmap[max_deg * 2], xxx_size, g.getNeighbor(kkk),
                      kkk_size, &vmap[max_deg]);
        if (thread_lane == 0)
          list_size[warp_lane][1] = local_count;
        __syncwarp();
        VID count = list_size[warp_lane][1];

#ifdef BITMAP_OPT
        if (count <= 32) {
          for (VID i = 0; i < count; i++) {
            auto search = g.getNeighbor(vmap[max_deg + i]);
            VID search_size = g.getOutDegree(vmap[max_deg + i]);
            if (thread_lane == 0)
              bitmap32[warp_lane][i] = 0;
            __syncwarp();
            for (auto j = thread_lane; j < count; j += WARP_SIZE) {
              unsigned active = __activemask();
              bool flag = (j != i) &&
                          binary_search(search, vmap[max_deg + j], search_size);
              __syncwarp(active);
              unsigned mask = __ballot_sync(active, flag);
              if (thread_lane == 0)
                bitmap32[warp_lane][i] = mask;
            }
          }
          for (VID i = thread_lane; i < count; i += 32) {
            VID v2 = vmap[max_deg + i];
            auto to_check = bitmap32[warp_lane][i];
            for (int j = 0; j < 32; j++) {
              if (to_check & (1 << j)) {
                counter += __popc(bitmap32[warp_lane][j] & to_check);
              }
            }
          }
        }

        else if (count <= 64) {
          for (VID i = 0; i < count; i++) {
            auto search = g.getNeighbor(vmap[max_deg + i]);
            VID search_size = g.getOutDegree(vmap[max_deg + i]);
            if (thread_lane == 0) {
              bitmap64[warp_lane][i * 2 + 0] = 0;
              bitmap64[warp_lane][i * 2 + 1] = 0;
            }
            __syncwarp();

            if (thread_lane < count) {
              unsigned active = __activemask();
              auto j = thread_lane;
              bool flag = (j != i) &&
                          binary_search(search, vmap[max_deg + j], search_size);
              __syncwarp(active);
              unsigned mask = __ballot_sync(active, flag);
              if (thread_lane == 0)
                bitmap64[warp_lane][i * 2 + 0] = mask;
            }
            __syncwarp();
            if (thread_lane + 32 < count) {
              unsigned active = __activemask();
              auto j = thread_lane + 32;
              bool flag = (j != i) &&
                          binary_search(search, vmap[max_deg + j], search_size);
              __syncwarp(active);
              unsigned mask = __ballot_sync(active, flag);
              if (thread_lane == 0)
                bitmap64[warp_lane][i * 2 + 1] = mask;
            }
            __syncwarp();
          }
          __syncwarp();

          if (thread_lane < count) {

            VID v2 = vmap[max_deg + thread_lane];
            auto to_check0 = bitmap64[warp_lane][thread_lane * 2 + 0];
            auto to_check1 = bitmap64[warp_lane][thread_lane * 2 + 1];
            for (int j = 0; j < 32; j++) {
              if (to_check0 & (1 << j)) {
                counter += __popc(bitmap64[warp_lane][j * 2 + 0] & to_check0);
                counter += __popc(bitmap64[warp_lane][j * 2 + 1] & to_check1);
              }
              if (to_check1 & (1 << j)) {
                counter +=
                    __popc(bitmap64[warp_lane][(j + 32) * 2 + 0] & to_check0);
                counter +=
                    __popc(bitmap64[warp_lane][(j + 32) * 2 + 1] & to_check1);
              }
            }
          }
          __syncwarp();
          if (thread_lane + 32 < count) {
            VID v2 = vmap[max_deg + thread_lane + 32];
            auto to_check0 = bitmap64[warp_lane][(thread_lane + 32) * 2 + 0];
            auto to_check1 = bitmap64[warp_lane][(thread_lane + 32) * 2 + 1];
            for (int j = 0; j < 32; j++) {
              if (to_check0 & (1 << j)) {
                counter += __popc(bitmap64[warp_lane][j * 2 + 0] & to_check0);
                counter += __popc(bitmap64[warp_lane][j * 2 + 1] & to_check1);
              }
              if (to_check1 & (1 << j)) {
                counter +=
                    __popc(bitmap64[warp_lane][(j + 32) * 2 + 0] & to_check0);
                counter +=
                    __popc(bitmap64[warp_lane][(j + 32) * 2 + 1] & to_check1);
              }
            }
          }
        } else
#endif
        // continue;
        {
          for (VID i = 0; i < count; i++) {
            auto search = g.getNeighbor(vmap[max_deg + i]);
            VID search_size = g.getOutDegree(vmap[max_deg + i]);
            for (auto j = thread_lane; j < count; j += WARP_SIZE) {
              unsigned active = __activemask();
              bool flag = (j != i) &&
                          binary_search(search, vmap[max_deg + j], search_size);
              __syncwarp(active);
              adj_lists.warp_set(offset, i, j, flag);
            }
          }
          // auto count = construct_induced_graph(g, v0, v1, adj_lists, vlists);
          __syncwarp();
          auto nc = (count - 1) / 32 + 1;
          for (VID v2 = 0; v2 < count; v2++) {
            for (VID v3 = 0; v3 < count; v3++) {
              if (adj_lists.get(offset, v2, v3)) {
                auto count1 = adj_lists.intersect_num(offset, nc, v2, v3);
                // if (thread_lane == 0) {
                // if (count1 > 0) printf("v0=%d, v1=%d, v2=%d, v3=%d,
                // count=%d\n", v0, v1, vmap[v2], vmap[v3], count1);
                counter += count1;
                //}
              }
            }
          }
        }
      }
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  atomicAdd(total, counter);
}

template <typename VID, typename VLABEL>
__global__ void clique4_graphfold(VID ne, dev::Graph<VID, VLABEL> g,
                                  VID *vlists, size_t max_deg, AccType *total,
                                  AccType *INDEX) {

  int thread_id = TID_1D;
  int warp_id = thread_id / WARP_SIZE; // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg)];
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK];
  for (VID eid = warp_id; eid < ne;) {
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
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  atomicAdd(total, counter);
}
