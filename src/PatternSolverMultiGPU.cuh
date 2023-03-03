#include "src/graph/operations.cuh"

// #include "omp.h"
using namespace project_GraphFold;

#include "src/gpu_kernels/pattern_kernels.cuh"
#include "src/utils/cuda_utils.h"
#include "src/utils/sm_pattern.h"
#include "src/utils/utils.h"

template <typename VID, typename VLABEL>
void PatternSolverMultiGPU(project_GraphFold::Graph<VID, VLABEL> &hg,
                           uint64_t &result, int n_dev,
                           project_GraphFold::modes cal_m,
                           project_GraphFold::Pattern p,
                           std::vector<VID> tasks) {
  
  VID ne = hg.get_enum();
  VID nv = hg.get_vnum();
  VID max_degree = hg.getMaxDegree();

  int count_length = 6;
  std::vector<AccType> h_total(n_dev * count_length, 0);
  std::vector<AccType *> d_total(n_dev);
  for (int i = 0; i < n_dev; ++i) {
    SetDevice(i);
    DMALLOC(d_total[i], sizeof(AccType) * count_length);
    TODEV(d_total[i], &h_total[i * count_length],
          sizeof(AccType) * count_length);
    WAIT();
    // std::cout << " host total ini" << i << " :" << h_total[i]<< std::endl;
  }
  WAIT();
  std::vector<std::thread> threads;
  std::vector<double> subt_cost(n_dev);
  double t_cost = 0;
  // TODEV(d_total[i], &h_total, sizeof(AccType));
  t_cost = wtime();
  using device_t = dev::Graph<VID, VLABEL>;
  for (int i = 0; i < n_dev; i++) {
    threads.push_back(std::thread([&, i]() {
      SetDevice(i);
      device_t d_g = hg.DeviceObject(i); // vector of pointers
      subt_cost[i] = wtime();
      int grid_size, block_size;

      int list_num = 8;
      size_t per_block_vlist_size =
          // WARPS_PER_BLOCK * size_t(k - 3) * size_t(max_degree) * sizeof(VID);
          WARPS_PER_BLOCK * list_num * size_t(max_degree) * sizeof(VID);

      uint32_t *matrix;
      int width = 1000;
      int block_per_v = (nv + width - 1) / width;
      int bitset_per_v = (block_per_v + 32 - 1) / 32;
      int global_idx = 0;
      long long col_len = (long long)(bitset_per_v);
      long long row_len = (long long)(nv);
      {
        if (p.get_name() == "P1") {
          if (hg.query_dense_graph())
            H_ERR(cudaOccupancyMaxPotentialBlockSize(
                &grid_size, &block_size, P1_unfused_matching<VID, VLABEL>, 0,
                (int)MAX_BLOCK_SIZE));
          else
            H_ERR(cudaOccupancyMaxPotentialBlockSize(
                &grid_size, &block_size, P1_count_correction<VID, VLABEL>, 0,
                // P1_unfused_matching<VID, VLABEL>, 0,
                (int)MAX_BLOCK_SIZE));
        } else if (p.get_name() == "P2") {
          H_ERR(cudaOccupancyMaxPotentialBlockSize(
              &grid_size, &block_size, P2_fused_matching<VID, VLABEL>, 0,
              // P2_unfused_matching<VID, VLABEL>, 0,
              (int)MAX_BLOCK_SIZE));
        } else if (p.get_name() == "P3") {
          if (hg.query_dense_graph())
            H_ERR(cudaOccupancyMaxPotentialBlockSize(
                &grid_size, &block_size, P3_unfused_matching<VID, VLABEL>, 0,

                (int)MAX_BLOCK_SIZE));
          else
            H_ERR(cudaOccupancyMaxPotentialBlockSize(
                &grid_size, &block_size, P3_fused_matching<VID, VLABEL>, 0,
                (int)MAX_BLOCK_SIZE));
        } else if (p.get_name() == "P4") {
          H_ERR(cudaOccupancyMaxPotentialBlockSize(
              &grid_size, &block_size, P4_unfused_matching<VID, VLABEL>, 0,
              (int)MAX_BLOCK_SIZE));
        } else if (p.get_name() == "P5") {
          H_ERR(cudaOccupancyMaxPotentialBlockSize(
              &grid_size, &block_size, P5_fused_matching<VID, VLABEL>, 0,
              (int)MAX_BLOCK_SIZE));
        } else if (p.get_name() == "P6") {
          if (hg.query_dense_graph())
            H_ERR(cudaOccupancyMaxPotentialBlockSize(
                &grid_size, &block_size,
                // P6_fused_matching<VID, VLABEL>, 0,
                P6_unfused_matching<VID, VLABEL>, 0, (int)MAX_BLOCK_SIZE));
          else
            H_ERR(cudaOccupancyMaxPotentialBlockSize(
                &grid_size, &block_size, P6_fused_matching<VID, VLABEL>, 0,
                (int)MAX_BLOCK_SIZE));
        } else if (p.get_name() == "P7") {
          if (hg.query_dense_graph())
            H_ERR(cudaOccupancyMaxPotentialBlockSize(
                &grid_size, &block_size, P7_unfused_matching<VID, VLABEL>, 0,
                (int)MAX_BLOCK_SIZE));
          else
            H_ERR(cudaOccupancyMaxPotentialBlockSize(
                &grid_size, &block_size, P7_fused_matching<VID, VLABEL>, 0,
                (int)MAX_BLOCK_SIZE));
        } else if (p.get_name() == "P8") {
          H_ERR(cudaOccupancyMaxPotentialBlockSize(
              &grid_size, &block_size, P8_fused_matching<VID, VLABEL>, 0,
              (int)MAX_BLOCK_SIZE));
        } else {
          std::cout << "User should develop it followed by G2Miner."
                    << std::endl;
          exit(-1);
        }
      }

#ifdef BLOCK_OPT

      long long nv_nv = (long long)((long long)row_len * (long long)col_len);
      uint32_t *h_matrix = (uint32_t *)malloc(sizeof(uint32_t) * nv_nv);

      DMALLOC(matrix, sizeof(uint32_t) * nv_nv);
      CLEAN(matrix, sizeof(uint32_t) * nv_nv);
      {
        for (int v = 0; v < nv; v += 1) {
          auto v_ptr = hg.getNeighbor(v);
          auto v_size = hg.getOutDegree(v);
          auto v_blk = v / width;
          auto v_blk_id = v_blk / 32;
          auto v_blk_oft = v_blk & 31;
          int pre = -1;
          int local_idx = 0;
          for (int j = 0; j < v_size; j++) {
            auto u = v_ptr[j];
            auto u_blk = u / width;
            auto u_oft = u % width;

            if (u_blk != pre) {
              local_idx++;
              pre = u_blk;
            }

            auto u_blk_id = u_blk / 32;
            auto u_blk_oft = u_blk & 31;
            long long addr0 =
                (long long)(v) * (long long)col_len + (long long)(u_blk_id);
            long long addr1 =
                (long long)(u) * (long long)col_len + (long long)(v_blk_id);
            h_matrix[addr0] |= (1 << u_blk_oft);
            h_matrix[addr1] |= (1 << v_blk_oft);
          }
        }
      }
      TODEV(matrix, h_matrix, sizeof(uint32_t) * nv_nv);
      printf("create matrix done!!\n");
      std::cout << "matrix  size: " << nv_nv * sizeof(uint32_t) / (1024 * 1024)
                << " MB\n";

#endif
      AccType *G_INDEX, *G_INDEX1, *G_INDEX2;
      AccType nowindex = (grid_size * block_size / WARP_SIZE);
      DMALLOC(G_INDEX, sizeof(AccType));
      DMALLOC(G_INDEX1, sizeof(AccType));
      DMALLOC(G_INDEX2, sizeof(AccType));
      TODEV(G_INDEX, &nowindex, sizeof(AccType));
      TODEV(G_INDEX1, &nowindex, sizeof(AccType));
      TODEV(G_INDEX2, &nowindex, sizeof(AccType));
      size_t flist_size = grid_size * per_block_vlist_size;
      std::cout << "flist_size is " << flist_size / (1024 * 1024)
                << " MB, grid_size is " << grid_size
                << ", per_block_vlist_size is " << per_block_vlist_size
                << std::endl;
      VID *d_frontier_list;
      int *freq_list;
      DMALLOC(d_frontier_list, flist_size);

      subt_cost[i] = wtime();
      if (p.get_name() == "P1") {

        if (hg.query_dense_graph())
          P1_unfused_matching<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX,
              matrix, width, col_len);
        else {
          P1_frequency_count<VID, VLABEL><<<grid_size, block_size>>>(
              nv, d_g, d_frontier_list, max_degree, d_total[i], G_INDEX);

          P1_count_correction<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX2);
        }
      } else if (p.get_name() == "P2") {
        P2_fused_matching<VID, VLABEL><<<grid_size, block_size>>>(
            tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX);

      } else if (p.get_name() == "P3") {
        if (hg.query_dense_graph())
          P3_unfused_matching<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX,
              matrix, width, col_len);
        else
          P3_fused_matching<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX);

      } else if (p.get_name() == "P4") {
        P4_unfused_matching<VID, VLABEL><<<grid_size, block_size>>>(
            tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX,
            matrix, width, col_len);
      } else if (p.get_name() == "P5") {
        P5_fused_matching<VID, VLABEL><<<grid_size, block_size>>>(
            tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX);
      } else if (p.get_name() == "P6") {
        if (hg.query_dense_graph())
          P6_unfused_matching<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX,
              matrix, width, col_len);
        else
          P6_fused_matching<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX,
              matrix, width, col_len);

      } else if (p.get_name() == "P7") {
        if (hg.query_dense_graph())
          P7_unfused_matching<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX,
              matrix, width, col_len);
        else
          P7_fused_matching<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX);

      } else if (p.get_name() == "P8") {
        P8_fused_matching<VID, VLABEL><<<grid_size, block_size>>>(
            tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX);
      } else {
        std::cout << "User should develop it followed by G2Miner." << std::endl;
        exit(-1);
      }
      WAIT();
      subt_cost[i] = wtime() - subt_cost[i];
      TOHOST(d_total[i], &h_total[i * count_length],
             sizeof(AccType) * count_length);
      WAIT();
    }));
  }

  for (auto &thread : threads)
    thread.join();
  WAIT();
  if (p.get_name() == "P4" || p.get_name() == "P6") {
    for (int i = 0; i < n_dev; ++i) {
      std::cout << p.get_name() << " from GPU[" << i << "] is "
                << h_total[i * count_length + 1] << std::endl;
      result += h_total[i * count_length + 1];
    }
    std::cout << "Result: " << result << "\n" << std::endl;
  } else if (p.get_name() == "P1") {
    AccType result0 = 0, result1 = 0;
    for (int i = 0; i < n_dev; ++i) {
      result0 = h_total[i * count_length + 1];
      result1 += h_total[i * count_length + 2];
    }
    std::cout << "Result: " << result0 - result1 << "\n" << std::endl;
  } else {
    for (int i = 0; i < n_dev; ++i) {
      auto val = h_total[i * count_length + 1] - h_total[i * count_length + 2];
      std::cout << p.get_name() << " from GPU[" << i << "] is " << val
                << std::endl;
      result += val;
    }
    std::cout << "Result: " << result << "\n" << std::endl;
  }

  for (int i = 0; i < n_dev; ++i)
    std::cout << "Kernel Time[GPU" << i << "] = " << subt_cost[i] << " seconds"
              << std::endl;

  t_cost = wtime() - t_cost;

}