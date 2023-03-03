#include "src/graph/scheduler.h"
namespace project_GraphFold {

template <typename VID, typename VLABEL>
std::vector<VID> Scheduler<VID, VLABEL>::round_robin(
    int n, project_GraphFold::Graph<VID, VLABEL> &hg,
    std::vector<VID *> &src_ptrs, std::vector<VID *> &dst_ptrs, int stride) {
  auto nnz = hg.getNNZ();
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
      if ((total_num_chunks % n == 0) && (chunk_id == nchunks_per_queue - 1) &&
          (qid == n - 1))
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

} // namespace project_GraphFold