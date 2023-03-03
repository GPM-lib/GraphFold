#include "src/graph/operations.cuh"

// #include "omp.h"
using namespace project_GraphFold;

#include "src/gpu_kernels/clique_kernels.cuh"

template <typename VID, typename VLABEL>
void CFSolver(project_GraphFold::Graph<VID, VLABEL> &hg, int k,
              uint64_t &result, int n_dev, project_GraphFold::modes cal_m) {
  ASSERT(k > 3); // guarantee clique > 4
  
  VID ne = hg.get_enum();
  VID max_degree = hg.getMaxDegree();

  size_t per_block_vlist_size =
      WARPS_PER_BLOCK * size_t(k - 3) * size_t(max_degree) * sizeof(VID);

  AccType *d_total;
  AccType h_total = 0;
  DMALLOC(d_total, sizeof(AccType));
  TODEV(d_total, &h_total, sizeof(AccType));
  WAIT();
  auto d_g = hg.DeviceObject();
  int grid_size, block_size; // uninitialized
  // residency_strategy

  if (k == 4) {
    H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                             clique4_graphfold<VID, VLABEL>, 0,
                                             (int)MAX_BLOCK_SIZE));
  } else if (k == 5) {
    H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                             clique5_graphfold<VID, VLABEL>, 0,
                                             (int)MAX_BLOCK_SIZE));
  } else if (k == 6) {
    H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                             clique6_graphfold<VID, VLABEL>, 0,
                                             (int)MAX_BLOCK_SIZE));
  } else if (k == 7) {
    H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                             clique7_graphfold<VID, VLABEL>, 0,
                                             (int)MAX_BLOCK_SIZE));
  }
  MultiBitsets<> adj_lists(grid_size * WARPS_PER_BLOCK, max_degree, max_degree);
  AccType *G_INDEX, *G_INDEX1, *G_INDEX2;
  AccType nowindex = (grid_size * block_size / WARP_SIZE);
  DMALLOC(G_INDEX, sizeof(AccType));
  DMALLOC(G_INDEX1, sizeof(AccType));
  DMALLOC(G_INDEX2, sizeof(AccType));
  TODEV(G_INDEX, &nowindex, sizeof(AccType));
  TODEV(G_INDEX1, &nowindex, sizeof(AccType));
  TODEV(G_INDEX2, &nowindex, sizeof(AccType));

  double start = wtime();
  size_t flist_size = grid_size * per_block_vlist_size;
  std::cout << "flist_size is " << flist_size / (1024 * 1024)
            << " MB, grid_size is " << grid_size << ", per_block_vlist_size is "
            << per_block_vlist_size << std::endl;
  VID *d_frontier_list;
  DMALLOC(d_frontier_list, flist_size);

  if (cal_m == e_centric) {
    if (k == 4) {
      clique4_graphfold<VID, VLABEL><<<grid_size, block_size>>>(
          ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX);
    } else if (k == 5) {
      // clique_5_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
      //     ne, d_g, d_frontier_list, max_degree, d_total);

      clique5_graphfold<VID, VLABEL><<<grid_size, block_size>>>(
          ne, d_g, d_frontier_list, adj_lists, max_degree, d_total, G_INDEX);
    } else if (k == 6) {
      // clique_5_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
      //     ne, d_g, d_frontier_list, max_degree, d_total);

      clique6_graphfold<VID, VLABEL><<<grid_size, block_size>>>(
          ne, d_g, d_frontier_list, adj_lists, max_degree, d_total, G_INDEX);
    } else if (k == 7) {
      // clique_5_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
      //     ne, d_g, d_frontier_list, max_degree, d_total);

      clique7_graphfold<VID, VLABEL><<<grid_size, block_size>>>(
          ne, d_g, d_frontier_list, adj_lists, max_degree, d_total, G_INDEX);
    } else {
      std::cout << "User should develop it followed by G2Miner." << std::endl;
      exit(-1);
    }
  } else if (cal_m == v_centric) {
    // vertex_centric
  } else {
    std::cout << "Wrong Calculation Mode." << std::endl;
  }
  WAIT();
  double end = wtime();
  std::cout << "CF" << k << " matching  time: " << (end - start) << " seconds"
            << std::endl;

  TOHOST(d_total, &h_total, sizeof(AccType));
  result = h_total;
  FREE(d_total);
}