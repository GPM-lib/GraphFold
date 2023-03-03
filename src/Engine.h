#pragma once
#include "graph/graph.h"
// #include "utils/utils.h"
// #include "src/utils/cuda_utils.h"
#include "CFSolver.cuh"
#include "G2MinerSolver.cuh"
#include "MCSolver.cuh"
#include "PatternSolver.cuh"
#include "TCSolver.cuh"
#include "src/utils/sm_pattern.h"

#include "CFSolverMultiGPU.cuh"
#include "G2MinerSolverMultiGPU.cuh"
#include "MCSolverMultiGPU.cuh"
#include "PatternSolverMultiGPU.cuh"
#include "TCSolverMultiGPU.cuh"

#include "src/graph/scheduler.h"
//#define EVEN_SPLIT_EDGE

#define RR

namespace project_GraphFold {
template <typename VID, typename VLABEL> class Engine {
private:
  Graph<VID, VLABEL> &hg_; // data graph
  Graph<VID, VLABEL> qg_;  // query graph ***const
  uint64_t result;
  int n_devices = 1;
  algorithms algo;
  modes cal_mode; // vertex, edge
  // NOTE: Mix used for MotifCounting and CliqueCounting
  int motif_k = 0;
  std::vector<uint64_t> motif_result; // WARNING: uninitialized
  project_GraphFold::Pattern s_pattern;

  int chunksize = 1024; // not specified
  std::vector<VID> tasks;

public:
  Engine(project_GraphFold::Graph<VID, VLABEL> &hg, int n_dev, algorithms al,
         project_GraphFold::modes cal_m)
      : hg_(hg), n_devices(n_dev), algo(al), cal_mode(cal_m) {
    tasks.resize(n_dev);
  }
  Engine(project_GraphFold::Graph<VID, VLABEL> &hg, int n_dev, algorithms al,
         project_GraphFold::modes cal_m, int k)
      : hg_(hg), n_devices(n_dev), algo(al), cal_mode(cal_m) {
    motif_k = k;
    tasks.resize(n_dev);
  } // motif & clique
  Engine(project_GraphFold::Graph<VID, VLABEL> &hg, int n_dev, algorithms al,
         project_GraphFold::modes cal_m, project_GraphFold::Pattern p)
      : hg_(hg), n_devices(n_dev), algo(al), cal_mode(cal_m) {
    tasks.resize(n_dev);
    s_pattern = p; // pattern
  }                // subgraph matching

  // Note: remind to reconstruct, algo is expired now.
  uint64_t RunTC() {
    // orientation here
    if (cal_mode == v_centric) {
      hg_.orientation(false);
      hg_.SortCSRGraph(true);
    } else
      hg_.orientation();
    hg_.copyToDevice(0, hg_.get_enum(), n_devices, false, false);
    TCSolver(hg_, result, n_devices, cal_mode);
    return result;
  }

  std::vector<uint64_t> RunMC() {
    motif_result = std::vector<uint64_t>(num_possible_patterns[motif_k], 0);
    hg_.copyToDevice(0, hg_.get_enum(), n_devices, false, false);
    MCSolver(hg_, motif_k, motif_result, n_devices, cal_mode);
    return motif_result;
  }

  uint64_t RunCF() {
    // orientation here
    hg_.orientation();
    hg_.copyToDevice(0, hg_.get_enum(), n_devices, false, false);
    CFSolver(hg_, motif_k, result, n_devices, cal_mode);
    return result;
  }

  uint64_t RunPatternEnum() {
    hg_.copyToDevice(0, hg_.get_enum(), n_devices, false, false);
    PatternSolver(hg_, result, n_devices, cal_mode, s_pattern);
    return result;
  }

  uint64_t RunG2Miner() {
    if (s_pattern.get_name() == "CF4" || s_pattern.get_name() == "CF5" ||
        s_pattern.get_name() == "CF6" || s_pattern.get_name() == "CF7" ||
        s_pattern.get_name() == "TC")
      hg_.orientation();
    hg_.copyToDevice(0, hg_.get_enum(), n_devices, false, false);
    G2MinerSolver(hg_, result, n_devices, cal_mode, s_pattern);
    return result;
  }

  uint64_t RunTCMultiGPU() {
    result = 0;
    // orientation here
    if (cal_mode == v_centric) {
      hg_.orientation(false);
      hg_.SortCSRGraph(true);
    } else {
      hg_.orientation();
    }
    // split the edgelist onto multi-gpus (scheduler strategies)
    TASK_SPLIT();

    // TCSolver(hg_, result, n_devices, cal_mode);
    TCSolverMultiGPU(hg_, result, n_devices, cal_mode, tasks);
    return result;
  }

  uint64_t RunCFMultiGPU() {
    result = 0;
    // orientation here
    hg_.orientation();
    // split the edgelist onto multi-gpus (scheduler strategies)
    TASK_SPLIT();

    CFSolverMultiGPU(hg_, motif_k, result, n_devices, cal_mode, tasks);
    return result;
  }
  uint64_t RunG2MinerMultiGPU() {
    result = 0;
    if (s_pattern.get_name() == "CF4" || s_pattern.get_name() == "CF5" ||
        s_pattern.get_name() == "CF6" || s_pattern.get_name() == "CF7" ||
        s_pattern.get_name() == "TC")
      hg_.orientation();

    TASK_SPLIT();
    G2MinerSolverMultiGPU(hg_, result, n_devices, cal_mode, s_pattern, tasks);
    return result;
  }

  uint64_t RunPatternEnumMultiGPU() {
    result = 0;
    TASK_SPLIT();
    PatternSolverMultiGPU(hg_, result, n_devices, cal_mode, s_pattern, tasks);
    return result;
  }

  void RunMCMultiGPU() {
    result = 0;
    TASK_SPLIT();
    motif_result = std::vector<uint64_t>(num_possible_patterns[motif_k], 0);
    MCSolverMultiGPU(hg_, motif_k, motif_result, n_devices, cal_mode, tasks);
  }

  //~Engine()
};
} // namespace project_GraphFold
