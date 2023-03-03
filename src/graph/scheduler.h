#ifndef GRAPH_SCHEDULER_H_
#define GRAPH_SCHEDULER_H_
#include "src/graph/graph.h"
#include <algorithm>

#define EVEN_SPLIT_EDGE()                                                      \
  std::cout << "Using multi-GPU partition stragegy: EVEN_SPLIT_EDGE."          \
            << std::endl;                                                      \
  VID num_tasks_per_gpu = CEIL_DIV(hg_.get_enum(), n_devices);                 \
  for (int i = 0; i < n_devices; ++i) {                                        \
    tasks[i] = num_tasks_per_gpu;                                              \
    printf("device:%d  task[%d]:%d\n", i, tasks[i]);                           \
  }                                                                            \
  tasks[n_devices - 1] = hg_.get_enum() - (n_devices - 1) * num_tasks_per_gpu; \
  hg_.copyToDevice(0, hg_.get_enum(), n_devices, false, false);                \
  for (int i = 0; i < n_devices; i++) {                                        \
    std::cout << "GPU " << i << " task " << tasks[i] << std::endl;             \
  }

#define ROUND_ROBIN()                                                          \
  std::cout << "Using multi-GPU partition stragegy: ROUND_ROBIN."              \
            << std::endl;                                                      \
  std::vector<VID *> src_ptrs, dst_ptrs;                                       \
  Scheduler<VID, VLABEL> scheduler;                                            \
  int chunk_size = 1024;                                                       \
  auto tasks =                                                                 \
      scheduler.round_robin(n_devices, hg_, src_ptrs, dst_ptrs, chunk_size);   \
  hg_.copyToDevice(n_devices, tasks, src_ptrs, dst_ptrs, false);               \
  for (int i = 0; i < n_devices; i++) {                                        \
    std::cout << "GPU " << i << " task " << tasks[i] << std::endl;             \
  }

//#define EVENSPLIT
#ifdef EVENSPLIT
#define TASK_SPLIT() EVEN_SPLIT_EDGE()
#else
#define TASK_SPLIT() ROUND_ROBIN()
#endif



// #e
namespace project_GraphFold {
template <typename VID, typename VLABEL> class Scheduler {
public:
  Scheduler() : nnz(0) {}
  ~Scheduler() {}
  std::vector<VID> round_robin(int n, project_GraphFold::Graph<VID, VLABEL> &hg,
                               std::vector<VID *> &src_ptrs,
                               std::vector<VID *> &dst_ptrs, int stride) {
    // auto nnz = hg.getNNZ();
    auto nnz = hg.get_enum();
    assert(nnz > 8192); // if edgelist is too small, no need to split
    std::cout << "split edgelist with chunk size of " << stride
              << " using chunked round robin\n";
    // resize workload list to n
    src_ptrs.resize(n);
    dst_ptrs.resize(n);

    VID total_num_chunks = (nnz - 1) / stride + 1;
    VID nchunks_per_queue = total_num_chunks / n;
    // every GPU queue basic size(except the border): every chunksize(edges) *
    // n_chunks
    std::vector<VID> lens(n, stride * nchunks_per_queue);
    if (total_num_chunks % n != 0) {
      for (int i = 0; i < n; i++) {
        if (i + 1 == int(total_num_chunks % n)) {
          lens[i] += nnz % stride == 0 ? stride : nnz % stride;
        } else if (i + 1 < int(total_num_chunks % n)) {
          lens[i] += stride;
        }
      }
    } else {
      lens[n - 1] = lens[n - 1] + nnz % stride - stride;
    }

    for (int i = 0; i < n; i++) {
      src_ptrs[i] = new VID[lens[i]];
      dst_ptrs[i] = new VID[lens[i]];
    }

    auto src_list = hg.getSrcPtr(0);
    auto dst_list = hg.getDstPtr(0);
#pragma omp parallel for
    for (VID chunk_id = 0; chunk_id < nchunks_per_queue; chunk_id++) {
      VID begin = chunk_id * n * stride;
      for (int qid = 0; qid < n; qid++) {
        VID pos = begin + qid * stride;
        int size = stride;
        if ((total_num_chunks % n == 0) &&
            (chunk_id == nchunks_per_queue - 1) && (qid == n - 1))
          size = nnz % stride;
        std::copy(src_list + pos, src_list + pos + size,
                  src_ptrs[qid] + chunk_id * stride);
        std::copy(dst_list + pos, dst_list + pos + size,
                  dst_ptrs[qid] + chunk_id * stride);
      }
    }

    VID begin = nchunks_per_queue * n * stride;
    for (int i = 0; i < n; i++) {
      VID pos = begin + i * stride;
      if (i + 1 == int(total_num_chunks % n)) {
        std::copy(src_list + pos, src_list + nnz,
                  src_ptrs[i] + nchunks_per_queue * stride);
        std::copy(dst_list + pos, dst_list + nnz,
                  dst_ptrs[i] + nchunks_per_queue * stride);
      } else if (i + 1 < int(total_num_chunks % n)) {
        std::copy(src_list + pos, src_list + pos + stride,
                  src_ptrs[i] + nchunks_per_queue * stride);
        std::copy(dst_list + pos, dst_list + pos + stride,
                  dst_ptrs[i] + nchunks_per_queue * stride);
      }
    }
    return lens;
  }

  std::vector<size_t> sort_indexes(const std::vector<VID> &v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
  }

  // template <typename VID, typename VLABEL>
  std::vector<VID>
  adaptive_round_robin(int n, project_GraphFold::Graph<VID, VLABEL> &hg,
                       std::vector<std::vector<VID>> &src_ptrs,
                       std::vector<std::vector<VID>> &dst_ptrs, int stride) {
    auto nnz = hg.getNNZ();
    assert(nnz > 8192); // if edgelist is too small, no need to split
    std::cout << "split edgelist with chunk size of " << stride
              << " using chunked round robin\n";
    // resize workload list to n
    src_ptrs.resize(n);
    dst_ptrs.resize(n);
    VID total_num_chunks = (nnz - 1) / stride + 1;
    // every GPU queue basic size(except the border): every chunksize(edges) *
    // n_chunks
    std::vector<VID> lens(n, 0); // init lens

    std::vector<VID> chunk_workload(total_num_chunks, 0);

    auto src_list = hg.getSrcPtr(0);
    auto dst_list = hg.getDstPtr(0);
    // calculate every chunk's workload
    VID chunk_id = 0;
    while (chunk_id < total_num_chunks - 1) {
      size_t chunk_l_tmp = 0;
      int index = chunk_id * stride;
      for (VID i = 0; i < stride; ++i) {
        // getEdgeWorkload
        size_t load1 = hg.getOutDegree(src_list[index + i]);
        size_t load2 = hg.getOutDegree(dst_list[index + i]);
        chunk_l_tmp += load1;
        chunk_l_tmp += load2;
      }
      chunk_workload[chunk_id] = chunk_l_tmp;
      chunk_id++;
    }
    // the last chunk workload
    if (chunk_id == total_num_chunks - 1) {
      // workload num
      int index = chunk_id * stride;
      for (VID i = 0; i < nnz % stride; ++i) {
        size_t load1 = hg.getOutDegree(src_list[index + i]);
        size_t load2 = hg.getOutDegree(dst_list[index + i]);
        chunk_workload[chunk_id] += load1;
        chunk_workload[chunk_id] += load2;
      }
    }

    auto load_indices = sort_indexes(chunk_workload);

    // initialize queue workload
    std::vector<size_t> queue_workload(n, 0);
    for (VID cid = 0; cid < total_num_chunks; ++cid) {
      // get current minimize workload queue's index
      size_t minIndex = std::distance(
          queue_workload.begin(),
          std::min_element(queue_workload.begin(), queue_workload.end()));
      // copy current chunk to src_ptrs and dst_ptrs
      int chk_to_load_idx = load_indices[cid]; // chunk id to process
      int start_offset = chk_to_load_idx * stride;
      // copy elements in current chunk
      if (cid != total_num_chunks - 1) {
        for (int i = 0; i < stride; ++i) {
          // LOG(INFO) <<"Check 3.1 "<< src_list[start_offset] ;
          src_ptrs[minIndex].push_back(src_list[start_offset + i]);
          dst_ptrs[minIndex].push_back(dst_list[start_offset + i]);
        }
        // update lens
        lens[minIndex] += stride;
      } else {
        int end = nnz % stride;
        for (int i = 0; i < end; ++i) {
          src_ptrs[minIndex].push_back(src_list[start_offset + i]);
          dst_ptrs[minIndex].push_back(dst_list[start_offset + i]);
        }
        // update lens
        lens[minIndex] += end;
      }
      // update queue_workload
      queue_workload[minIndex] += chunk_workload[chk_to_load_idx];
    }
    // print lens
    for (int i = 0; i < n; ++i) {
      std::cout << "edge in GPU " << i << " is " << lens[i] << std::endl;
    }
    return lens;
  }

private:
  std::vector<VID> srcs;
  std::vector<VID> dsts;
  VID nnz; // number of tasks/edges

};

} // namespace project_GraphFold
#endif