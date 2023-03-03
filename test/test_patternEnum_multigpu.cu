#include <iostream>

#include "src/Engine.h"
#include "src/graph/graph.h"
#include "src/graph/io.h"

using namespace project_GraphFold;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <graph_path> <pattern_name> <num_gpus>." << std::endl;
    return 1;
  }
  std::string path = std::string(argv[1]);
  project_GraphFold::Pattern pattern(argv[2]);
  std::cout << "Pattern: " << pattern.get_name()
            << "  matching using undirected graphs."
            << "\n"
            << std::endl;
  Loader<int, int> loader;
  Graph<int, int> hg; // data graph
  algorithms algo = CF;
  modes cal_mode = e_centric; // TODO: support more cal_mode.
  uint64_t result = 0;
  int n_devices = atoi(argv[3]);

  // load data graph
  loader.Load(path, "mtx");
  loader.Build(hg, n_devices);

  Engine<int, int> engine(hg, n_devices, algo, cal_mode, pattern);
  result = engine.RunPatternEnumMultiGPU();

  // std::cout << "Result: " << result << "\n"<<std::endl;
  std::cout << "Pattern: " << pattern.get_name() << " matching done."
            << std::endl;
  return 0;
}