// #include "src/utils/timer.h"
#include "src/graph/operations.cuh"
//#include "src/utils/launcher.h"
using namespace project_GraphFold;

#include "src/gpu_kernels/tc_bs_warp_edge.cuh"
#include "src/gpu_kernels/tc_hi_warp_vertex.cuh"
#include "src/gpu_kernels/tuning_schedules.cuh"
#include "src/utils/sm_pattern.h"

template <typename VID, typename VLABEL>
void TCSolver(project_GraphFold::Graph<VID, VLABEL> &hg, uint64_t &result,
              int n_dev, project_GraphFold::modes cal_m) {
  
  // size_t nblocks = MAX_GRID_SIZE;
  VID ne = hg.get_enum();
  VID nv = hg.get_vnum();
  AccType *d_total;
  AccType h_total = 0;
  DMALLOC(d_total, sizeof(AccType));
  TODEV(d_total, &h_total, sizeof(AccType));
  WAIT();
  auto d_g = hg.DeviceObject();
  double time_cost = 0;
  if (cal_m == e_centric) {
    double start = wtime();

    int grid_size, block_size;
        H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                 tc_warp_edge<VID, VLABEL>, 0,
                                                 (int)MAX_BLOCK_SIZE));
        tc_warp_edge<<<grid_size, block_size>>>(ne, d_g,
                                                d_total); // ne
        WAIT();
    double end = wtime();
    time_cost = (end - start);
  } else if (cal_m == v_centric) {
    size_t nthreads = 1024;
    size_t nblocks = 1024;
    size_t bucketnum = BLOCK_BUCKET_NUM;
    size_t bucketsize = TR_BUCKET_SIZE;

    VID *bins;
    auto bins_mem = nblocks * bucketnum * bucketsize * sizeof(VID);
    H_ERR(cudaMalloc((void **)&bins, bins_mem));
    H_ERR(cudaMemset(bins, 0, bins_mem));

    int nowindex[3];
    int *G_INDEX;

    int block_range = 0;
    // if vertex's degree is large than threshold(USE_CTA), use block to process
    if (true) {
      int l = 0, r = nv;
      int val = USE_CTA;
      while (l < r - 1) {
        int mid = (l + r) / 2;
        if (hg.edge_begin(mid + 1) - hg.edge_begin(mid) > val)
          l = mid;
        else
          r = mid;
      }
      if (hg.edge_begin(l + 1) - hg.edge_begin(l) <= val)
        block_range = 0;
      else
        block_range = l + 1;
    }
    nowindex[0] = nblocks * nthreads / WARP_SIZE;
    nowindex[1] = nblocks;
    nowindex[2] = block_range + (nblocks * nthreads / WARP_SIZE);
    H_ERR(cudaMalloc((void **)&G_INDEX, sizeof(int) * 3));
    H_ERR(cudaMemcpy(G_INDEX, &nowindex, sizeof(int) * 3,
                     cudaMemcpyHostToDevice));
    double start = wtime();
    tc_hi_warp_vertex<<<nblocks, nthreads>>>(nv, d_g, bins, d_total, G_INDEX,
                                             block_range);
    WAIT();
    double end = wtime();
    time_cost = end - start;
  } else {
    std::cout << "Wrong Calculation Mode." << std::endl;
  }
  std::cout << "Triangle counting  time: " << time_cost << " seconds"
            << std::endl;
  TOHOST(d_total, &h_total, sizeof(AccType));
  result = h_total;
  FREE(d_total);
}
