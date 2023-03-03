#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/graph/operations.cuh"
#include "src/graph/scan.h"
#include "src/utils/buffer.h"
#include "src/utils/cuda_utils.h"
#include "src/utils/logging.h"
#include "src/utils/timer.h"
#include "src/utils/utils.h"

namespace project_GraphFold {

// VID
template <typename VID, typename VLABEL> class Graph;

namespace dev {
template <typename VID, typename VLABEL> class Graph {
private:
  size_t vsize_;  // uninitialized
  size_t esize_;  // uninitialized
  VID max_degree; // uninitialized
  Buffer<VLABEL> vlabels_;
  Buffer<VID> row_ptr_;
  Buffer<VID> col_idx_;
  Buffer<VID> odegs_;
  Buffer<VID> src_list_;
  Buffer<VID> dst_list_;

  template <typename _VID, typename _VLABEL>
  friend class project_GraphFold::Graph;

public:
  // Graph(project_GraphFold::Graph& hg) { init(hg);}
  VID get_vnum() const { return vsize_; }
  VID get_enum() const { return esize_; }
  DEV_INLINE VID get_src(VID edge) const { return src_list_.data()[edge]; }
  DEV_INLINE VID get_dst(VID edge) const { return dst_list_.data()[edge]; }
  DEV_INLINE VID getOutDegree(VID src) {
    return col_idx_.data()[src + 1] - col_idx_.data()[src];
  } // check
  DEV_INLINE size_t get_colidx_size() const { return col_idx_.size(); }
  DEV_INLINE VID edge_begin(VID src) const { return col_idx_.data()[src]; }
  DEV_INLINE VID edge_end(VID src) const { return col_idx_.data()[src + 1]; }
  DEV_INLINE VID get_edge_dst(VID idx) const { return row_ptr_.data()[idx]; }
  // Test and dump COO
  DEV_INLINE void DumpCO() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("Dump COO: src_list size: %d, dst_list size: %d.\n",
             src_list_.size(), dst_list_.size());
      for (int i = 0; i < src_list_.size(); i++) {
        printf("%d ", src_list_.data()[i]);
      }
      printf("\n");
      for (int i = 0; i < dst_list_.size(); i++) {
        printf("%d ", dst_list_.data()[i]);
      }
    }
  }

  DEV_INLINE VID *getNeighbor(VID vid) const {
    return const_cast<VID *>(row_ptr_.data()) + col_idx_.data()[vid];
  }
};
} // namespace dev

template <typename VID, typename VLABEL> class Graph {
public:
  using device_t = dev::Graph<VID, VLABEL>;
  // TODO: To support multiple partition in vertex-cut manner.
  // To this end, we have to store the vertex mapping(original_id->local_id)
  // get the neighborlist START pointer
  VID *getNeighbor(VID vid) const {
    return const_cast<VID *>(row_ptr_.data()) + col_idx_.data()[vid];
  }
  VID edge_begin(VID src) const { return col_idx_.data()[src]; }
  VID edge_end(VID src) const { return col_idx_.data()[src + 1]; }

  VID get_src(VID idx) const { return src_list_[idx]; }
  VID get_dst(VID idx) const { return dst_list_[idx]; }

  VID get_vnum() const { return vsize_; }
  VID get_enum() const { return esize_; }
  VID getMaxDegree() { return max_degree_; }
  VID *getSrcPtr(VID start) { return src_list_.data() + start; }
  VID *getDstPtr(VID start) { return dst_list_.data() + start; }
  VID getOutDegree(VID src) {
    return col_idx_.data()[src + 1] - col_idx_.data()[src];
  }
  size_t getNNZ() { return nnz; }
  VID CalMaxDegree(std::vector<VID> out_degs) {
    auto maxPosition = max_element(out_degs.begin(), out_degs.end());
    return *maxPosition;
  }

  bool graphPropertyAnalyze(std::vector<VID> out_degs) {
    // auto maxPosition = min_element(out_degs.begin(), out_degs.end());

    if (out_degs.size() >= 3000000)
      return false;
    double sum = std::accumulate(out_degs.begin(), out_degs.end(), 0.0);
    double mean = sum / out_degs.size(); //均值

    double accum = 0.0;
    std::for_each(std::begin(out_degs), std::end(out_degs),
                  [&](const double d) { accum += (d - mean) * (d - mean); });

    double stdev = sqrt(accum / (out_degs.size() - 1)); //方差
    if (mean > 20 && mean < 40)
      if (stdev > 20 && stdev < 40) {
        return true;
      }
    return false;
  }
  // USE_DAG on with orientation
  void orientation(bool NeedToLoadToDevice = true) {
    std::cout << "Orientation enabled, DAG generated.\n" << std::endl;
    double start = wtime();
    std::vector<VID> new_odegs_(vsize_, 0);
#pragma omp parallel for
    // Dump(std::cout);
    for (VID src = 0; src < vsize_; ++src) {
      VID *neighlist = getNeighbor(src);
      Buffer<VID> tmp(neighlist, getOutDegree(src));
      // std::cout << " size of neighlist: " << sizeof(neighlist);
      for (auto dst : tmp) {
        // std::cout << "i is " << i << ", dst is " << dst;
        if (odegs_[dst] > odegs_[src] ||
            (odegs_[dst] == odegs_[src] && dst > src)) {
          new_odegs_[src]++;
        }
      }
    }

    VID new_max_degree_ = CalMaxDegree(new_odegs_);
    std::cout << "Orientation Generating: New max degree is: "
              << new_max_degree_ << std::endl;
    // vector type: this.row_ptr_; this.col_idx_;
    std::vector<VID> new_row_ptr_;
    std::vector<VID> new_col_idx_;
    std::vector<VID> new_src_list_;
    new_col_idx_.resize(vsize_ + 1);
    parallel_prefix_sum<VID, VID>(new_odegs_,
                                  new_col_idx_.data()); // vector satisfied
    auto n_edges_ = new_col_idx_[vsize_];
    new_row_ptr_.resize(n_edges_);
    new_src_list_.resize(n_edges_);
#pragma omp parallel for
    for (VID src = 0; src < vsize_; ++src) {
      VID *neighlist = getNeighbor(src);
      Buffer<VID> tmp(neighlist, getOutDegree(src));
      auto begin = new_col_idx_[src];
      VID offset = 0;
      for (auto dst : tmp) {
        if (odegs_[dst] > odegs_[src] ||
            (odegs_[dst] == odegs_[src] && dst > src)) {
          new_row_ptr_[begin + offset] = dst;
          new_src_list_[begin + offset] = src;
          offset++;
        }
      }
    }
    // Update graph info
    row_ptr_ = new_row_ptr_;
    col_idx_ = new_col_idx_;
    esize_ = n_edges_;
    max_degree_ = new_max_degree_;

    double end = wtime();
    std::cout << "Orientation Generating time: " << (end - start) << " seconds"
              << std::endl;

    src_list_ = new_src_list_;
    dst_list_ = new_row_ptr_;
  
  }

  void SortCSRGraph(bool NeedToLoadToDevice = true) {
    std::vector<int> index(vsize_);
    std::vector<int> r_index(vsize_);
    for (int i = 0; i < index.size(); i++)
      index[i] = i;
    std::stable_sort(index.begin(), index.end(), [&](int a, int b) {
      return getOutDegree(a) > getOutDegree(b);
    });

    std::vector<int> new_col_idx_(vsize_ + 1);
    std::vector<int> new_row_ptr_(esize_);
    std::vector<VID> new_odegs_(vsize_, 0);

    for (VID src = 0; src < vsize_; src++) {
      VID v = index[src];
      r_index[v] = src;
    }

    for (VID src = 0; src < vsize_; src++) {
      VID v = index[src];
      new_odegs_[src] = getOutDegree(v);
    }
    parallel_prefix_sum<VID, VID>(new_odegs_,
                                  new_col_idx_.data()); // vector satisfied
    for (VID src = 0; src < vsize_; src++) {
      VID v = index[src];
      VID *neighlist = getNeighbor(v);
      Buffer<VID> tmp(neighlist, getOutDegree(v));
      auto begin = new_col_idx_[src];
      VID offset = 0;
      for (auto dst : tmp) {
        new_row_ptr_[begin + offset] = r_index[dst];
        offset++;
      }
      std::sort(&new_row_ptr_[begin], &new_row_ptr_[begin + offset]);
    }

    col_idx_ = new_col_idx_;
    row_ptr_ = new_row_ptr_;
    odegs_ = new_odegs_;
  }

  // initialize the size of device pointer vector
  void resizeDeviceVector(int n_dev) {
    d_row_ptr_.resize(n_dev);
    d_odegs_.resize(n_dev);
    d_col_idx_.resize(n_dev);
    d_src_list_.resize(n_dev);
    d_dst_list_.resize(n_dev);
    d_vlabels_.resize(n_dev);
  }

  void copyToDevice(size_t start, size_t end, int n_dev, bool sym_break = false,
                    bool use_label = false) {
    resizeDeviceVector(n_dev);
    auto n = end - start;
    VID n_tasks_per_gpu = (n - 1) / n_dev + 1;
    for (int i = 0; i < n_dev; ++i) {
      SetDevice(i);
      if (use_label) {
        d_vlabels_[i].resize(vsize_);
        TODEV(thrust::raw_pointer_cast(d_vlabels_.data()), vlabels_.data(),
              sizeof(VLABEL) * vsize_);
      }

      VID begin = start + i * n_tasks_per_gpu;
      // Note: Test only.
      // if (!sym_break) d_dst_list_[i] = row_ptr_.data() + begin;
      VID num = n_tasks_per_gpu;
      if (begin + num > end)
        num = end - begin; // begin is the index for copy starting
      // initialize CSR
      d_row_ptr_[i].resize(esize_);
      d_odegs_[i].resize(vsize_);
      d_col_idx_[i].resize(vsize_ + 1);
      // initialize COO task list with size 'num'
      d_src_list_[i].resize(num);
      d_dst_list_[i].resize(num);
      // copy all CSR
      TODEV(thrust::raw_pointer_cast(d_row_ptr_[i].data()), row_ptr_.data(),
            sizeof(VID) * esize_);
      TODEV(thrust::raw_pointer_cast(d_odegs_[i].data()), odegs_.data(),
            sizeof(VID) * vsize_);
      TODEV(thrust::raw_pointer_cast(d_col_idx_[i].data()), col_idx_.data(),
            sizeof(VID) * (vsize_ + 1)); // size_ to VID
      // copy partial
      TODEV(thrust::raw_pointer_cast(d_src_list_[i].data()),
            src_list_.data() + begin, sizeof(VID) * num);
      if (!sym_break) {
        TODEV(thrust::raw_pointer_cast(d_dst_list_[i].data()),
              row_ptr_.data() + begin, sizeof(VID) * num);
      } else {
        TODEV(thrust::raw_pointer_cast(d_dst_list_[i].data()),
              dst_list_.data() + begin, sizeof(VID) * num);
      } // sym_break_copy

      WAIT();
      std::cout << "Successful fill into GPU[" << i << "]." << std::endl;
    }
  }

  void copyToDevice(int n_dev, std::vector<VID> tasks, std::vector<VID *> &srcs,
                    std::vector<VID *> &dsts, bool use_label = false) {
    // Timer t;
    // t.Start();
    resizeDeviceVector(n_dev);
    for (int i = 0; i < n_dev; ++i) {
      SetDevice(i);
      if (use_label) {
        d_vlabels_[i].resize(vsize_);
        TODEV(thrust::raw_pointer_cast(d_vlabels_.data()), vlabels_.data(),
              sizeof(VLABEL) * vsize_);
      }
      // initialize CSR
      d_row_ptr_[i].resize(esize_);
      d_odegs_[i].resize(vsize_);
      d_col_idx_[i].resize(vsize_ + 1);

      // initialize COO task list with size 'num'
      auto num = tasks[i];
      d_src_list_[i].resize(num);
      d_dst_list_[i].resize(num);

      // copy all CSR
      TODEV(thrust::raw_pointer_cast(d_row_ptr_[i].data()), row_ptr_.data(),
            sizeof(VID) * esize_);
      TODEV(thrust::raw_pointer_cast(d_odegs_[i].data()), odegs_.data(),
            sizeof(VID) * vsize_);
      TODEV(thrust::raw_pointer_cast(d_col_idx_[i].data()), col_idx_.data(),
            sizeof(VID) * (vsize_ + 1)); // size_ to VID

      // copy partial
      VID *src_ptr = srcs[i];
      VID *dst_ptr = dsts[i];

      // printf("hi, %d",int(srcs[i]));
      TODEV(thrust::raw_pointer_cast(d_src_list_[i].data()), src_ptr,
            sizeof(VID) * num); // srcs[i]
      TODEV(thrust::raw_pointer_cast(d_dst_list_[i].data()), dst_ptr,
            sizeof(VID) * num); // dsts[i]
      WAIT();
      std::cout << "Successful fill into GPU[" << i << "]." << std::endl;
    }
  }

  void copyToDevice(int n_dev, std::vector<VID> tasks,
                    std::vector<std::vector<VID>> &srcs,
                    std::vector<std::vector<VID>> &dsts,
                    bool use_label = false) {
    resizeDeviceVector(n_dev);
    for (int i = 0; i < n_dev; ++i) {
      SetDevice(i);
      if (use_label) {
        d_vlabels_[i].resize(vsize_);
        TODEV(thrust::raw_pointer_cast(d_vlabels_.data()), vlabels_.data(),
              sizeof(VLABEL) * vsize_);
      }
      // initialize CSR
      d_row_ptr_[i].resize(esize_);
      d_odegs_[i].resize(vsize_);
      d_col_idx_[i].resize(vsize_ + 1);

      // initialize COO task list with size 'num'
      auto num = tasks[i];
      d_src_list_[i].resize(num);
      d_dst_list_[i].resize(num);

      // copy all CSR
      TODEV(thrust::raw_pointer_cast(d_row_ptr_[i].data()), row_ptr_.data(),
            sizeof(VID) * esize_);
      TODEV(thrust::raw_pointer_cast(d_odegs_[i].data()), odegs_.data(),
            sizeof(VID) * vsize_);
      TODEV(thrust::raw_pointer_cast(d_col_idx_[i].data()), col_idx_.data(),
            sizeof(VID) * (vsize_ + 1)); // size_ to VID

      // copy partial
      VID *src_ptr = srcs[i].data();
      VID *dst_ptr = dsts[i].data();

      // printf("hi, %d",int(srcs[i]));
      TODEV(thrust::raw_pointer_cast(d_src_list_[i].data()), src_ptr,
            sizeof(VID) * num); // srcs[i]?
      TODEV(thrust::raw_pointer_cast(d_dst_list_[i].data()), dst_ptr,
            sizeof(VID) * num); // dsts[i]?
      WAIT();
      std::cout << "Successful fill into GPU[" << i << "]." << std::endl;
    }
  }


  // TODO: Need to be polished, remove tasks_v, no long needed.
  void WorkLoadCalulator(int n_dev, std::vector<VID> &tasks,
                         std::vector<VID> &tasks_v) {
    std::cout << "Calulating workload vertex info for GPUs..." << std::endl;
    std::vector<VID> start(
        n_dev, 0); // refer to the start pointer of src_list and dst_list
    for (int i = 1; i < n_dev; ++i) {
      start[i] = start[i - 1] + tasks[i - 1];
    }
    std::vector<std::vector<VID>> v_tmp(n_dev);
    // v_tmp[i].insert(v_tmp[i].end(), hg.src_list_.begin() + start[i],
    // hg.src_list_.begin() + start[i+1]); v_tmp[i].insert(v_tmp[i].end(),
    // hg.dst_list_.begin() + start[i], hg.dst_list_.begin() + start[i+1]);
    for (int i = 0; i < n_dev; ++i) {
      tasks_v[i] = 0; //
      for (int j = 0; j < tasks[i]; ++j) {
        VID p = src_list_[start[i] + j];
        VID q = dst_list_[start[i] + j];
        if (!(std::find(v_tmp[i].begin(), v_tmp[i].end(), p) !=
              v_tmp[i].end())) {
          v_tmp[i].push_back(p);
          tasks_v[i]++;
        }
        if (!(std::find(v_tmp[i].begin(), v_tmp[i].end(), q) !=
              v_tmp[i].end())) {
          v_tmp[i].push_back(q);
          tasks_v[i]++;
        }
      }
    }
  }

  void Init(std::vector<VID> const &vids, std::vector<VLABEL> const &vlabels,
            std::vector<std::pair<VID, VID>> const &edges, int n_dev,
            bool use_label = false) {
    std::cout << "Initializing graph..." << std::endl;
    double start = wtime();

    vsize_ = vids.size();
    esize_ = edges.size();
    if (use_label)
      vlabels_ = std::move(vlabels);
    odegs_.resize(vsize_);
    col_idx_.resize(vsize_ + 1);
    row_ptr_.resize(esize_);

    src_list_.resize(esize_);
    dst_list_.resize(esize_);

    for (size_t i = 0; i < edges.size(); ++i) {
      odegs_[edges[i].first]++;
    }

    col_idx_[0] = 0;
    for (size_t i = 0; i < vsize_; ++i) {
      col_idx_[i + 1] = col_idx_[i] + odegs_[i];
      odegs_[i] = 0;
    }

    // directed edges
    for (size_t i = 0; i < esize_; ++i) {
      VID v0 = edges[i].first;
      VID v1 = edges[i].second;
      row_ptr_[col_idx_[v0] + odegs_[v0]] = v1;
      odegs_[v0]++;
    }

    double end = wtime();
    std::cout << "CSR transforming time: " << end - start << "s" << std::endl;
    std::cout << " -- vsize: " << vsize_ << " esize: " << esize_ << "\n"
              << std::endl;
    // calculate max degree
    max_degree_ = CalMaxDegree(odegs_); // VID

    is_dense_graph = graphPropertyAnalyze(odegs_); // VID

    // generating COO
    // Note: May use vector<std::pair> instead.
    double start_coo = wtime();
    nnz = esize_; // no sym_break, no ascend.
    for (size_t i = 0; i < esize_; ++i) {
      src_list_[i] = edges[i].first;
      dst_list_[i] = edges[i].second;
    }
    double end_coo = wtime();
    std::cout << "COO loading time: " << end_coo - start_coo << "s"
              << std::endl;

  }

  // Only for single GPU
  device_t DeviceObject() const {
    device_t dg;
    // if (use_label)
    // dg.vlabels_ = Buffer<VLABEL>(d_vlabels_[0]);
    dg.row_ptr_ = Buffer<VID>(d_row_ptr_[0]);
    dg.odegs_ = Buffer<VID>(d_odegs_[0]);
    dg.col_idx_ = Buffer<VID>(d_col_idx_[0]);
    dg.src_list_ = Buffer<VID>(d_src_list_[0]);
    dg.dst_list_ = Buffer<VID>(d_dst_list_[0]);

    return dg;
  }

  device_t DeviceObject(int dev_id,
                        bool use_label = false) const { // DEV_HOST, now is HOST
    device_t dg;
    if (use_label)
      dg.vlabels_ = Buffer<VLABEL>(d_vlabels_[dev_id]);
    dg.row_ptr_ = Buffer<VID>(d_row_ptr_[dev_id]);
    dg.odegs_ = Buffer<VID>(d_odegs_[dev_id]);
    dg.col_idx_ = Buffer<VID>(d_col_idx_[dev_id]);
    dg.src_list_ = Buffer<VID>(d_src_list_[dev_id]);
    dg.dst_list_ = Buffer<VID>(d_dst_list_[dev_id]);


    return dg;
  }

  void Dump(std::ostream &out) {
    out << "vsize: " << vsize_ << " esize: " << esize_ << "\n";
    out << "labels: ";
    for (size_t i = 0; i < vsize_; ++i) {
      out << vlabels_[i] << " ";
    }
    out << "\n";
    out << "row_ptr: ";
    for (size_t i = 0; i < esize_; ++i) {
      out << row_ptr_[i] << " ";
    }
    out << "\n";
    out << "col_idx: ";
    for (size_t i = 0; i < vsize_ + 1; ++i) {
      out << col_idx_[i] << " ";
    }
    out << "\n";
  }

  void DumpCOO(std::ostream &out) {
    out << "vsize: " << vsize_ << " esize: " << esize_ << "\n";
    out << "labels: ";
    for (size_t i = 0; i < vsize_; ++i) {
      out << vlabels_[i] << " ";
    }
    out << "\n";
    out << "src_list: ";
    for (size_t i = 0; i < esize_; ++i) {
      out << src_list_[i] << " ";
    }
    out << "\n";
    out << "dst_list: ";
    for (size_t i = 0; i < esize_; ++i) {
      out << dst_list_[i] << " ";
    }
    out << "\n";
  }

  bool query_dense_graph() { return is_dense_graph; }

private:
  // Warning: NOT support device_id & n_gpu yet.
  size_t fid_; // ?
  size_t vsize_;
  size_t esize_;
  bool is_dense_graph;
  VID max_degree_;
  std::vector<VLABEL> vlabels_;

  // int num_vertex_classes; // label classes count
  // may used by filter
  // std::vector<VID> labels_frequency_;
  // VID max_label_frequency_;
  // int max_label;
  // std::vector<nlf_map> nlf_;
  // std::vector<VID> sizes;
  // CSR
  std::vector<VID> row_ptr_;
  std::vector<VID> col_idx_;
  std::vector<VID> odegs_; // <size_t>
  // add elabels_
  // COO
  VID nnz;
  std::vector<VID> src_list_; // <size_t> thrust host vector?
  std::vector<VID> dst_list_; // <size_t>

  // Warning: More supported format may increase the storage.
  // Every GPU has its device vector.
  std::vector<thrust::device_vector<VLABEL>> d_vlabels_;
  std::vector<thrust::device_vector<VID>> d_row_ptr_;
  std::vector<thrust::device_vector<VID>> d_odegs_;
  std::vector<thrust::device_vector<VID>> d_col_idx_;
  // assign tasks
  std::vector<thrust::device_vector<VID>> d_src_list_;
  std::vector<thrust::device_vector<VID>> d_dst_list_;
};

} // namespace project_GraphFold

#endif // endif
