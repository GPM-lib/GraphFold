#include "omp.h"
#include "src/graph/operations.cuh"

using namespace project_GraphFold;

#include "src/gpu_kernels/motif_kernels.cuh"
#include "src/gpu_kernels/pattern_kernels.cuh"
#include "src/utils/cuda_utils.h"
#include "src/utils/sm_pattern.h"
#include "src/utils/utils.h"

template <typename VID, typename VLABEL>
void MCSolver(project_GraphFold::Graph<VID, VLABEL> &hg, int motif_k,
              std::vector<uint64_t> &result, int n_dev,
              project_GraphFold::modes cal_m) {
  ASSERT(motif_k >= 3); // guarantee motif

  
  VID ne = hg.get_enum();
  VID nv = hg.get_vnum();
  size_t num_patterns = result.size();
  std::cout << "num patterns" << num_patterns << std::endl;
  AccType *h_accumulators = (AccType *)malloc(sizeof(AccType) * num_patterns);
  for (int i = 0; i < num_patterns; ++i)
    h_accumulators[i] = 0;

  AccType *d_accumulators;
  DMALLOC(d_accumulators, sizeof(AccType) * num_patterns);
  CLEAN(d_accumulators, sizeof(AccType) * num_patterns);


  auto d_g = hg.DeviceObject();
  // Calculate and allocate the frontiers
  // size_t nlists = (motif_k == 3) ? 0 : 2;
  size_t nlists = (motif_k == 3) ? 0 : 4;
  // Warning: cal max_degree
  VID max_degree = hg.getMaxDegree();
  size_t per_block_vlist_size =
      WARPS_PER_BLOCK * nlists * size_t(max_degree) * sizeof(VID);
  std::cout << "WARPS_PER_BLOCK is " << WARPS_PER_BLOCK << " nlists " << nlists
            << " size_t(max_degree) " << size_t(max_degree) << "sizeof(VID)"
            << sizeof(VID) << std::endl;
  // Note: grid_size i.e. nblocks
  int grid_size, block_size;
  // ???
  // add sth. for motif-4 block counting
  if (motif_k == 4) {
    H_ERR(cudaOccupancyMaxPotentialBlockSize(
        &grid_size, &block_size,
        motif4_warp_edge_bsopt_part_woPATH4_fusion<VID, VLABEL>, 0,
        (int)MAX_BLOCK_SIZE));
  }

  size_t flist_size = grid_size * per_block_vlist_size;
  std::cout << "flist_size is " << flist_size / (1024 * 1024)
            << " MB, grid_size is" << grid_size << ", per_block_vlist_size is "
            << per_block_vlist_size << std::endl;
  VID *d_frontier_list; // only for motif-4
  DMALLOC(d_frontier_list, flist_size);

  AccType *G_INDEX0, *G_INDEX1, *G_INDEX2, *G_INDEX3;

  int nowindex = (grid_size * block_size / WARP_SIZE);

  auto n_size = grid_size * WARPS_PER_BLOCK * size_t(max_degree);
  int *int_map;
  DMALLOC(int_map, n_size * sizeof(VID));

  DMALLOC(G_INDEX0, sizeof(AccType));
  DMALLOC(G_INDEX1, sizeof(AccType));
  DMALLOC(G_INDEX2, sizeof(AccType));
  DMALLOC(G_INDEX3, sizeof(AccType));
  TODEV(G_INDEX0, &nowindex, sizeof(AccType));
  TODEV(G_INDEX1, &nowindex, sizeof(AccType));
  TODEV(G_INDEX2, &nowindex, sizeof(AccType));
  TODEV(G_INDEX3, &nowindex, sizeof(AccType));

  uint32_t *matrix;
  int width = 1000;
  int block_per_v = (nv + width - 1) / width;
  int bitset_per_v = (block_per_v + 32 - 1) / 32;
  int global_idx = 0;
  long long col_len = (long long)(bitset_per_v);
  long long row_len = (long long)(nv);

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

  // Preprocess  Get each vertex's neighbours information about <disconnectivity
  // pair> number
  VID *count_list;
  DMALLOC(count_list, sizeof(VID) * nv);

  double start = wtime();
  if (cal_m == e_centric) {
    if (motif_k == 4) {

      P1_frequency_count<VID, VLABEL><<<grid_size, block_size>>>(
          nv, d_g, d_frontier_list, max_degree, d_accumulators, G_INDEX0);
      P1_count_correction<VID, VLABEL><<<grid_size, block_size>>>(
          ne, d_g, d_frontier_list, max_degree, d_accumulators, G_INDEX1);
      sub<<<1, 1>>>(d_accumulators);
      cudaDeviceSynchronize();
      motif4_warp_edge_bsopt_part_woPATH4_fusion<VID, VLABEL>
          <<<grid_size, block_size>>>(ne, d_g, d_frontier_list, max_degree,
                                      d_accumulators, G_INDEX2, int_map);
      P4_unfused_matching<VID, VLABEL><<<grid_size, block_size>>>(
          ne, d_g, d_frontier_list, max_degree, d_accumulators, G_INDEX3,
          matrix, width, col_len);
      cudaDeviceSynchronize();

      WAIT(); 
    } else {
      std::cout << "User should develop it followed by G2Miner." << std::endl;
      exit(-1);
    }
  } else if (cal_m == v_centric) {
  } else {
    std::cout << "Wrong Calculation Mode.\n" << std::endl;
    exit(-1);
  }
  double end = wtime();
  std::cout << "Motif counting  time: " << (end - start) << " seconds"
            << std::endl;

  TOHOST(d_accumulators, h_accumulators, sizeof(AccType) * num_patterns);

  for (size_t i = 0; i < num_patterns; ++i)
    result[i] = h_accumulators[i];
  FREE(d_accumulators);
}