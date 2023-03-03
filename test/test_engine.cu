#include <iostream>

#include "src/Engine.h"
#include "src/graph/graph.h"
#include "src/graph/io.h"
// #include "src/utils/cuda_utils.h"

using namespace project_GraphFold;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <graph_path> <graph_label>"
              << std::endl;
    return 1;
  }
  std::string path = std::string(argv[1]);
  std::string label_path = std::string(argv[2]); // label --

  Loader<int, int> loader;
  Graph<int, int> hg; // data graph
  algorithms algo = TC;
  modes cal_mode = e_centric; // TODO: support more cal_mode.
  uint64_t result = 0;
  int n_devices = 1; // TODO: support more devices.

  // load data graph
  loader.Load(path, "mtx");
  loader.LoadVLabel(label_path); // label_path --
  loader.Build(hg);

  Engine<int, int> engine(hg, n_devices, algo, cal_mode);
  result = engine.RunTC();

  std::cout << "Result: " << result << "\n" << std::endl;
  std::cout << "Engine test done." << std::endl;

  return 0;
}