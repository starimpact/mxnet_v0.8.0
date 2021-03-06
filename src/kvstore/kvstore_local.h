/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_local.h
 * @brief  local implementation
 */
#ifndef MXNET_KVSTORE_KVSTORE_LOCAL_H_
#define MXNET_KVSTORE_KVSTORE_LOCAL_H_

#include <mxnet/kvstore.h>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <utility>
#include <algorithm>
#include "./comm.h"

namespace mxnet {
namespace kvstore {
/**
 * \brief store data in local machine
 */
class KVStoreLocal : public KVStore {
 public:
  /*
   * \param use_device_comm
   */
  explicit KVStoreLocal(bool use_device_comm) : KVStore() {
    if (use_device_comm) {
      comm_ = new CommDevice();
    } else {
      comm_ = new CommCPU();
    }
    pinned_ctx_ = (MXNET_USE_CUDA != 0) ? Context::CPUPinned(0) : Context::CPU();
  }

  virtual ~KVStoreLocal() {
    delete comm_;
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(local_.find(keys[i]) == local_.end())
          << "duplicate init of key " << keys[i];
      local_[keys[i]] = values[i].Copy(pinned_ctx_);
      comm_->Init(keys[i], values[i].shape());
    }
  }

  // when do init, the ori_shapes is actually the partial shapes. This is not same to push_partial.
  // ori_indexes is keep the same size with the partial weights,
  // values is the original complete weights. This is not same to push_partial.
  void Init_Partial(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            const std::vector<TShape>& ori_shapes,
            const std::vector<Intlist>& ori_indexes) override {
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(local_.find(keys[i]) == local_.end())
          << "duplicate init of key " << keys[i];
      local_[keys[i]] = values[i].Copy(pinned_ctx_);
      local_partial_[keys[i]] = NDArray(ori_shapes[i], pinned_ctx_);

      for (int si = 0; si < partial_statenum; si++) {
        states_[keys[i]][si] = NDArray(values[i].shape(), pinned_ctx_);
        states_partial_[keys[i]][si] = NDArray(ori_shapes[i], pinned_ctx_);
      }
      comm_->Init(keys[i], ori_shapes[i]);
    }
  }

  void Push_Partial(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            const std::vector<TShape>& ori_shapes,
            const std::vector<Intlist>& ori_indexes,
            int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    std::vector<TShape> grouped_ori_shapes;
    std::vector<Intlist> grouped_ori_indexes;
    GroupKVPairs_Partial(keys, values, ori_shapes, ori_indexes,
        &uniq_keys, &grouped_vals, &grouped_ori_shapes, &grouped_ori_indexes);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
//      const TShape& ori_shape = grouped_ori_shapes[i];
      const Intlist& ori_index = grouped_ori_indexes[i];
      const NDArray& merged = comm_->Reduce(key, grouped_vals[i], priority);
      NDArray& local = local_[key];
      NDArray& local_partial = local_partial_[key];
      std::vector<NDArray>& states = states_[key];
      std::vector<NDArray>& states_partial = states_partial_[key];
      if (partial_updater_ != nullptr) {
        CHECK(!local.is_none()) << "key " << key << " has not been inited";
        // if merged is on gpu, we may need copy weight from cpu to gpu
        if (merged.ctx().dev_mask() != cpu::kDevMask &&
            local_partial.ctx().dev_mask() == cpu::kDevMask) {
          local_partial = local_partial.Copy(merged.ctx());
          for (int si = 0; si < partial_statenum; si++) {
            states_partial[si] = states_partial[si].Copy(merged.ctx());
          }
        }
        CopyFromTo_IndexFrom(local, &local_partial, ori_index, 0);
        for (int si = 0; si < partial_statenum; si++) {
          CopyFromTo_IndexFrom(states[si], &states_partial[si], ori_index, 0);
        }

        partial_updater_(key, merged, &local_partial, &states_partial);

        CopyFromTo_IndexTo(local_partial, &local, ori_index, 0);
        for (int si = 0; si < partial_statenum; si++) {
          CopyFromTo_IndexTo(states_partial[si], &states[si], ori_index, 0);
        }
      } else {
        CopyFromTo_IndexTo(merged, &local, ori_index, 0);
      }
    }
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const NDArray& merged = comm_->Reduce(key, grouped_vals[i], priority);
      NDArray& local = local_[key];
      if (updater_ != nullptr) {
        CHECK(!local.is_none()) << "key " << key << " has not been inited";
        // if merged is on gpu, we may need copy weight from cpu to gpu
        if (merged.ctx().dev_mask() != cpu::kDevMask &&
            local.ctx().dev_mask() == cpu::kDevMask) {
          local = local.Copy(merged.ctx());
        }
        updater_(key, merged,  &local);
      } else {
        local = merged;
      }
    }
  }

  void Pull_Partial(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            const std::vector<TShape>& ori_shapes,
            const std::vector<Intlist>& ori_indexes,
            int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    std::vector<TShape> grouped_ori_shapes;
    std::vector<Intlist> grouped_ori_indexes;
    GroupKVPairs_Partial(keys, values, ori_shapes, ori_indexes,
        &uniq_keys, &grouped_vals, &grouped_ori_shapes, &grouped_ori_indexes);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const Intlist& ori_index = grouped_ori_indexes[i];
      const NDArray& local = local_[key];
      CHECK(!local.is_none()) << "key " << key << " has not been inited";
      NDArray& local_partial = local_partial_[key];
      if (local_partial.shape()[0] == ori_index.size()) {
        CopyFromTo_IndexFrom(local, &local_partial, ori_index, 0);
        comm_->Broadcast(key, local_partial, grouped_vals[i], priority);
      } else if (local.shape()[0] == ori_index.size()) {
        comm_->Broadcast(key, local, grouped_vals[i], priority);
      } else {
        LG << "error: ori_index is wrong!\n";
      }
    }
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const NDArray& local = local_[key];
      CHECK(!local.is_none()) << "key " << key << " has not been inited";
      comm_->Broadcast(key, local, grouped_vals[i], priority);
    }
  }

 protected:
  /**
   * \brief group values on keys
   */
  template <typename V>
  void GroupKVPairs(const std::vector<int>& keys,
                    const std::vector<V>& values,
                    std::vector<int>* uniq_keys,
                    std::vector<std::vector<V> >* grouped_vals) {
    CHECK_EQ(keys.size(), values.size());
    // TODO(mli) check if already sorted as an optimization
    using Idx = std::pair<int, int>;
    std::vector<Idx> idx(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      idx[i].first = keys[i]; idx[i].second = i;
    }
    std::sort(idx.begin(), idx.end(), [](const Idx& a, const Idx& b) {
        return a.first < b.first;
      });

    int pre_key = idx[0].first - 1;
    for (auto i : idx) {
      if (i.first != pre_key) {
        uniq_keys->push_back(i.first);
        grouped_vals->push_back({values[i.second]});
        pre_key = i.first;;
      } else {
        grouped_vals->back().push_back(values[i.second]);
      }
    }
  }

  template <typename V>
  void GroupKVPairs_Partial(const std::vector<int>& keys,
                    const std::vector<V>& values,
                    const std::vector<TShape>& ori_shapes,
                    const std::vector<std::vector<int>>& ori_indexes,
                    std::vector<int>* uniq_keys,
                    std::vector<std::vector<V> >* grouped_vals,
                    std::vector<TShape>* grouped_ori_shapes,
                    std::vector<Intlist>* grouped_ori_indexes) {
    CHECK_EQ(keys.size(), values.size());
    // TODO(mli) check if already sorted as an optimization
    using Idx = std::pair<int, int>;
    std::vector<Idx> idx(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      idx[i].first = keys[i]; idx[i].second = i;
    }
    std::sort(idx.begin(), idx.end(), [](const Idx& a, const Idx& b) {
        return a.first < b.first;
      });

    int pre_key = idx[0].first - 1;
    for (auto i : idx) {
      if (i.first != pre_key) {
        uniq_keys->push_back(i.first);
        grouped_vals->push_back({values[i.second]});
        grouped_ori_shapes->push_back(ori_shapes[i.second]);
        grouped_ori_indexes->push_back(ori_indexes[i.second]);
        pre_key = i.first;
      } else {
        grouped_vals->back().push_back(values[i.second]);
      }
    }
  }

  /// reducer and broadcaster
  Comm* comm_;
  /// pinned context
  Context pinned_ctx_;
  /// \brief buffer for storing local values
  std::unordered_map<int, NDArray> local_;
  std::unordered_map<int, std::vector<NDArray> > states_;
  std::unordered_map<int, NDArray> local_partial_;
  std::unordered_map<int, std::vector<NDArray> > states_partial_;
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_LOCAL_H_
