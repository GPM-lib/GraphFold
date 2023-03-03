#include <iostream>

#include "src/Engine.h"
#include "src/graph/graph.h"
#include "src/graph/io.h"
#include "src/utils/sm_pattern.h"
using namespace project_GraphFold;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <graph_path> <n>" << std::endl;
    return 1;
  }
  std::string path = std::string(argv[1]);
  int k = atoi(argv[2]);
  int num_patterns = num_possible_patterns[k];   // the possible motif numß
  std::vector<uint64_t> result(num_patterns, 0); // initialization
  Loader<int, int> loader;
  Graph<int, int> hg; // data graph
  algorithms algo = MC;
  modes cal_mode = e_centric; // TODO: support more cal_mode.
  int n_devices = 1;          // TODO: support more devices.

  // load data graph
  loader.Load(path, "mtx");
  loader.Build(hg);

  Engine<int, int> engine(hg, n_devices, algo, cal_mode,
                          k); // specific args: k
  result = engine.RunMC();
  //  TODO: g.get_max_degree();
  std::cout << "Result: \n" << std::endl;
  std::cout << "Num of Patterns: " << num_patterns << "\n" << std::endl;
  for (int i = 0; i < num_patterns; i++) {
    std::cout << "Pattern " << i << ": " << result[i] << "\n";
  }
  std::cout << "MotifCounting test done." << std::endl;
  return 0;
}