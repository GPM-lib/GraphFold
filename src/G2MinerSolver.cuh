#include "src/graph/operations.cuh"

// #include "omp.h"
using namespace project_GraphFold;

#include "src/gpu_kernels/G2Miner_kernels.cuh"
#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"
#include "src/utils/sm_pattern.h"

template <typename VID, typename VLABEL>
void G2MinerSolver(project_GraphFold::Graph<VID, VLABEL> &hg, uint64_t &result,
                   int n_dev, project_GraphFold::modes cal_m,
                   project_GraphFold::Pattern p) {
  
  
  VID ne = hg.get_enum();
  VID nv = hg.get_vnum();
  VID max_degree = hg.getMaxDegree();

  int list_num = 6;
  size_t per_block_vlist_size =
      // WARPS_PER_BLOCK * size_t(k - 3) * size_t(max_degree) * sizeof(VID);
      WARPS_PER_BLOCK * list_num * size_t(max_degree) * sizeof(VID);

  AccType *d_total;
  int count_length = 6;
  AccType h_total[count_length] = {0};
  DMALLOC(d_total, count_length * sizeof(AccType));
  TODEV(d_total, &h_total, count_length * sizeof(AccType));
  // CLEAN(d_total, 4*sizeof(AccType));
  WAIT();
  auto d_g = hg.DeviceObject();
  int grid_size, block_size; // uninitialized

  {
    if (p.get_name() == "P1") {

      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P1_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "P2") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P2_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "P3") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P3_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "P4") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P4_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "P5") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P5_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "P6") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P6_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "P7") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P7_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "P8") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P8_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "CF4") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               clique4_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "CF5") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               clique5_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "CF6") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               clique6_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));

    } else if (p.get_name() == "CF7") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               clique7_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "TC") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               tc_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else if (p.get_name() == "MC4") {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               motif_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    } else {
      std::cout << "User should develop it followed by G2Miner." << std::endl;
      exit(-1);
    }
  }
  size_t flist_size = grid_size * per_block_vlist_size;
  std::cout << "flist_size is " << flist_size / (1024 * 1024)
            << " MB, grid_size is " << grid_size << ", per_block_vlist_size is "
            << per_block_vlist_size << std::endl;
  VID *d_frontier_list;
  DMALLOC(d_frontier_list, flist_size);

  double start = wtime();
  if (p.get_name() == "P1") {
    P1_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(ne, d_g, d_frontier_list,
                                                       max_degree, d_total);
  } else if (p.get_name() == "P2") {
    P2_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(ne, d_g, d_frontier_list,
                                                       max_degree, d_total);
  } else if (p.get_name() == "P3") {
    P3_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(ne, d_g, d_frontier_list,
                                                       max_degree, d_total);
  } else if (p.get_name() == "P4") {
    P4_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(ne, d_g, d_frontier_list,
                                                       max_degree, d_total);
  } else if (p.get_name() == "P5") {
    P5_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(ne, d_g, d_frontier_list,
                                                       max_degree, d_total);
  } else if (p.get_name() == "P6") {
    P6_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(ne, d_g, d_frontier_list,
                                                       max_degree, d_total);
  } else if (p.get_name() == "P7") {
    P7_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(ne, d_g, d_frontier_list,
                                                       max_degree, d_total);
  } else if (p.get_name() == "P8") {
    P8_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(ne, d_g, d_frontier_list,
                                                       max_degree, d_total);
  } else if (p.get_name() == "CF4") {
    clique4_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  } else if (p.get_name() == "CF5") {
    clique5_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  } else if (p.get_name() == "CF6") {
    clique6_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  } else if (p.get_name() == "CF7") {
    clique7_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  } else if (p.get_name() == "TC") {
    tc_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(ne, d_g, d_total);
  } else if (p.get_name() == "MC4") {
    motif_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }

  else {
    std::cout << "User should develop it followed by G2Miner." << std::endl;
    exit(-1);
  }

  WAIT();
  double end = wtime();
  std::cout << p.get_name() << " matching  time: " << (end - start)
            << " seconds" << std::endl;

  TOHOST(d_total, &h_total, count_length * sizeof(AccType));
  result = h_total[0];

  if (p.get_name() == "MC4") {
    for (int i = 0; i < 6; i++) {
      std::cout << "Pattern " << i << ": " << h_total[i] << "\n";
    }
  }

  FREE(d_total);
}