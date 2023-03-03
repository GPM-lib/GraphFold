#include <iostream>

#include "src/Engine.h"
#include "src/graph/graph.h"
#include "src/graph/io.h"
#include "src/utils/sm_pattern.h"
using namespace project_GraphFold;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <graph_path>"
              << " <num_gpus>." << std::endl;
    return 1;
  }
  std::string path = std::string(argv[1]);
  int num_patterns = 1;                          // the possible motif num
  std::vector<uint64_t> result(num_patterns, 0); // initialization
  Loader<int, int> loader;
  Graph<int, int> hg; // data graph
  algorithms algo = TC;
  // modes cal_mode = v_centric;  // TODO: support more cal_mode.
  modes cal_mode = e_centric;    // TODO: support more cal_mode.
  int n_devices = atoi(argv[2]); // TODO: support more devices.

  // load data graph
  loader.Load(path, "mtx");
  loader.Build(hg, n_devices);

  Engine<int, int> engine(hg, n_devices, algo, cal_mode); // specific args: k
  result[0] = engine.RunTCMultiGPU();

  std::cout << "TriangleCounting test done." << std::endl;
  return 0;
}