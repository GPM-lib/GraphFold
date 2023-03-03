#include "src/graph/operations.cuh"

// #include "omp.h"
using namespace project_GraphFold;

#include "src/gpu_kernels/clique_kernels.cuh"

template <typename VID, typename VLABEL>
void CFSolverMultiGPU(project_GraphFold::Graph<VID, VLABEL> &hg, int k,
                      uint64_t &result, int n_dev,
                      project_GraphFold::modes cal_m, std::vector<VID> tasks) {
  ASSERT(k > 3); // guarantee clique > 4
  
  
  VID ne = hg.get_enum();
  VID max_degree = hg.getMaxDegree();

  size_t per_block_vlist_size =
      WARPS_PER_BLOCK * size_t(k - 3) * size_t(max_degree) * sizeof(VID);


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
      int grid_size, block_size;

      if (k == 4) {
        H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                 clique4_graphfold<VID, VLABEL>,
                                                 0, (int)MAX_BLOCK_SIZE));
      } else if (k == 5) {
        H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                 clique5_graphfold<VID, VLABEL>,
                                                 0, (int)MAX_BLOCK_SIZE));
      } else if (k == 6) {
        H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                 clique6_graphfold<VID, VLABEL>,
                                                 0, (int)MAX_BLOCK_SIZE));
      } else if (k == 7) {
        H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                 clique7_graphfold<VID, VLABEL>,
                                                 0, (int)MAX_BLOCK_SIZE));
      }
      MultiBitsets<> adj_lists(grid_size * WARPS_PER_BLOCK, max_degree,
                               max_degree);
      AccType *G_INDEX, *G_INDEX1, *G_INDEX2;
      AccType nowindex = (grid_size * block_size / WARP_SIZE);
      DMALLOC(G_INDEX, sizeof(AccType));
      DMALLOC(G_INDEX1, sizeof(AccType));
      DMALLOC(G_INDEX2, sizeof(AccType));
      TODEV(G_INDEX, &nowindex, sizeof(AccType));
      TODEV(G_INDEX1, &nowindex, sizeof(AccType));
      TODEV(G_INDEX2, &nowindex, sizeof(AccType));

      subt_cost[i] = wtime();
      size_t flist_size = grid_size * per_block_vlist_size;
      std::cout << "flist_size is " << flist_size / (1024 * 1024)
                << " MB, grid_size is " << grid_size
                << ", per_block_vlist_size is " << per_block_vlist_size
                << std::endl;
      VID *d_frontier_list;
      DMALLOC(d_frontier_list, flist_size);

      if (cal_m == e_centric) {
        if (k == 4) {
          clique4_graphfold<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, max_degree, d_total[i], G_INDEX);
        } else if (k == 5) {
          clique5_graphfold<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, adj_lists, max_degree, d_total[i],
              G_INDEX);
        } else if (k == 6) {

          clique6_graphfold<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, adj_lists, max_degree, d_total[i],
              G_INDEX);
        } else if (k == 7) {

          clique7_graphfold<VID, VLABEL><<<grid_size, block_size>>>(
              tasks[i], d_g, d_frontier_list, adj_lists, max_degree, d_total[i],
              G_INDEX);
        } else {
          std::cout << "User should develop it followed by G2Miner."
                    << std::endl;
          exit(-1);
        }
      } else if (cal_m == v_centric) {
        // vertex_centric
      } else {
        std::cout << "Wrong Calculation Mode." << std::endl;
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
    std::cout << "CF from GPU[" << i << "] is " << h_total[i] << std::endl;
  for (int i = 0; i < n_dev; ++i)
    std::cout << "Kernel Time[GPU" << i << "] = " << subt_cost[i] << " seconds"
              << std::endl;

  for (int i = 0; i < n_dev; ++i)
    result += h_total[i];

  t_cost = wtime() - t_cost;
  std::cout << "Result: " << result << "\n" << std::endl;

}
