// #include "src/utils/timer.h"
#include "src/graph/operations.cuh"

using namespace project_GraphFold;

#include "src/gpu_kernels/tc_bs_warp_edge.cuh"
#include "src/gpu_kernels/tc_hi_warp_vertex.cuh"
#include "src/gpu_kernels/tuning_schedules.cuh"
#include "src/utils/sm_pattern.h"

template <typename VID, typename VLABEL>
void TCSolverMultiGPU(project_GraphFold::Graph<VID, VLABEL> &hg,
                      uint64_t &result, int n_dev,
                      project_GraphFold::modes cal_m, std::vector<VID> tasks) {
  // size_t nblocks = MAX_GRID_SIZE;
  VID ne = hg.get_enum();
  VID nv = hg.get_vnum();

  std::vector<AccType> h_total(n_dev, 0);
  std::vector<AccType *> d_total(n_dev);
  for (int i = 0; i < n_dev; ++i) {
    SetDevice(i);
    DMALLOC(d_total[i], sizeof(AccType));
    TODEV(d_total[i], &h_total[i], sizeof(AccType));
    WAIT();
    // std::cout << " host total ini" << i << " :" << h_total[i]<< std::endl;
  }
  WAIT();
  std::vector<std::thread> threads;
  std::vector<double> subt_cost(n_dev);
  double t_cost = 0;
  // TODEV(d_total, &h_total, sizeof(AccType));
  t_cost = wtime();
  using device_t = dev::Graph<VID, VLABEL>;
  for (int i = 0; i < n_dev; i++) {
    threads.push_back(std::thread([&, i]() {
      SetDevice(i);
      device_t d_g = hg.DeviceObject(i); // vector of pointers
      subt_cost[i] = wtime();
      // kernel launch
      if (cal_m == e_centric) {
        int grid_size, block_size;
        H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                 tc_warp_edge<VID, VLABEL>, 0,
                                                 (int)MAX_BLOCK_SIZE));
        // tc_warp_edge<VID, VLABEL>
        //     <<<grid_size, block_size>>>(tasks[i], d_g, d_total[i]); // ne
        tc_warp_edge<<<grid_size, block_size>>>(tasks[i], d_g,
                                                d_total[i]); // ne
        WAIT();
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
        // if vertex's degree is large than threshold(USE_CTA), use block to
        // process
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
        WAIT();
        tc_hi_warp_vertex<<<nblocks, nthreads>>>(nv, d_g, bins, d_total[i],
                                                 G_INDEX, block_range);
        WAIT();
      } else {
        std::cout << "Wrong Calculation Mode." << std::endl;
        exit(-1);
      }
      WAIT();
      subt_cost[i] = wtime() - subt_cost[i];
      TOHOST(d_total[i], &h_total[i], sizeof(AccType));
      WAIT();
    }));
  }

  for (auto &thread : threads)
    thread.join();
  WAIT();
  for (int i = 0; i < n_dev; ++i)
    std::cout << "Triangle from GPU[" << i << "] is " << h_total[i]
              << std::endl;
  for (int i = 0; i < n_dev; ++i)
    std::cout << "Kernel Time[GPU" << i << "] = " << subt_cost[i] << " seconds"
              << std::endl;

  for (int i = 0; i < n_dev; ++i)
    result += h_total[i];
  t_cost = wtime() - t_cost;
  std::cout << "Result: " << result << "\n" << std::endl;
  std::cout << "Triangle counting Total Time: " << t_cost << " seconds"
            << std::endl;
  // FREE(d_total);
}
