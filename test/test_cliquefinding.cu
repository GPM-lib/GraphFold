#include <iostream>

#include "src/Engine.h"
#include "src/graph/graph.h"
#include "src/graph/io.h"

using namespace project_GraphFold;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <graph_path> <n>" << std::endl;
    return 1;
  }
  std::string path = std::string(argv[1]);
  int k = atoi(argv[2]);
  std::cout << "K-clique listing using undirected graphs."
            << "\n"
            << std::endl;
  Loader<int, int> loader;
  Graph<int, int> hg; // data graph
  algorithms algo = CF;
  modes cal_mode = e_centric; // TODO: support more cal_mode.
  uint64_t result = 0;
  int n_devices = 1; // TODO: support more devices.

  // load data graph
  loader.Load(path, "mtx");
  loader.Build(hg);

  Engine<int, int> engine(hg, n_devices, algo, cal_mode, k);
  result = engine.RunCF();

  std::cout << "Result: " << result << "\n" << std::endl;
  std::cout << "K-clique counting test done." << std::endl;
  return 0;
}