/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_
#include <string>
#include <vector>
#include "./kvstore_local.h"
#include "mxnet/engine.h"
#include "ps/ps.h"
#include "./kvstore_dist_server.h"

namespace mxnet {
namespace kvstore {

/**
 * \brief distributed kvstore
 *
 * for a worker node, it always guarantees that all push and pull issued from
 * this worker on the same key are serialized. namely push(3) and then pull(3),
 * then the data pulled is always containing the modification from the push(3).
 *
 * it's the server node's job to control the data consistency among all
 * workers. see details on \ref ServerHandle::Start
 */
class KVStoreDist : public KVStoreLocal {
 public:
  explicit KVStoreDist(bool use_device_comm)
      : KVStoreLocal(use_device_comm), ps_worker_(nullptr), server_(nullptr) {
    if (IsWorkerNode()) {
      ps_worker_ = new ps::KVWorker<real_t>(0);
      ps::StartAsync("mxnet\0");
      if (!ps::Postoffice::Get()->is_recovery()) {
        ps::Postoffice::Get()->Barrier(
          ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
      }
    }
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);
  }

  virtual ~KVStoreDist() {
    Engine::Get()->WaitForAll();
    if (IsWorkerNode()) {
      if (barrier_before_exit_) {
        Barrier();
        if (get_rank() == 0) {
          // stop the executor at servers
          SendCommandToServers(kStopServer, "");
        }
      }
      ps::Finalize(barrier_before_exit_);
      delete ps_worker_;
    }
  }

  // when do init, the ori_shapes is actually the partial shapes. This is not same to push_partial.
  // ori_indexes is keep the same size with the partial weights,
  // values is the original complete weights. This is not same to push_partial.
  void Init_Partial(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            const std::vector<TShape>& ori_shapes,
            const std::vector<Intlist>& ori_indexes) override {
    CheckUnique(keys);
  //  std::cout << "Init_Partial" << std::endl;
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], ori_shapes[i]);
    }
    if (get_rank() == 0) {
      std::vector<TShape> ori_shapestmp(ori_shapes.size());
      for (size_t j = 0; j < ori_shapes.size(); j++) {
        ori_shapestmp[j] = TShape(values[j].shape());
      }
      Push_Partial_(keys, values, ori_shapestmp, ori_indexes, 0, false);
      // wait until the push is finished
      for (const auto& v : values) {
        v.WaitToWrite();
      }
    } else {
      // do nothing
    }
    if (!ps::Postoffice::Get()->is_recovery()) {
      Barrier();
    }
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], values[i].shape());
    }
    if (true && get_rank() == 0) {
      Push_(keys, values, 0, false);
      // wait until the push is finished
      for (const auto& v : values) {
        v.WaitToWrite();
      }
    } else {
      // do nothing
    }
    if (!ps::Postoffice::Get()->is_recovery()) {
      Barrier();
    }
  }

  void Push_Partial(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            const std::vector<TShape>& ori_shapes,
            const std::vector<Intlist>& ori_indexes,
            int priority) override {
 //   std::cout << "Push_Partial" << std::endl;
    Push_Partial_(keys, values, ori_shapes, ori_indexes, priority, true);
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    Push_(keys, values, priority, true);
  }

  void Pull_Partial(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            const std::vector<TShape>& ori_shapes,
            const std::vector<Intlist>& ori_indexes,
            int priority) override {
  //  std::cout << "Pull_Partial" << std::endl;
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    std::vector<TShape> grouped_ori_shapes;
    std::vector<Intlist> grouped_ori_indexes;
    GroupKVPairs_Partial(keys, values, ori_shapes, ori_indexes,
        &uniq_keys, &grouped_vals, &grouped_ori_shapes, &grouped_ori_indexes);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const auto& vals = grouped_vals[i];
      const TShape& ori_shape = grouped_ori_shapes[i];
      const Intlist& ori_index = grouped_ori_indexes[i];
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      NDArray recv_buf;
      auto& recv_buf0 = comm_buf_[key];
      if (recv_buf0.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf0 = NDArray(vals[0]->shape(), pinned_ctx_);
        recv_buf = recv_buf0;
      }  else {
        if (vals[0]->shape()[0] > recv_buf0.shape()[0]) {
          recv_buf0 = NDArray(vals[0]->shape(), pinned_ctx_);
        }
        CHECK(recv_buf0.shape()[0] >= vals[0]->shape()[0]) << "Pull_Partial: vals shape_0 must be LE than recv_buf0.";
        recv_buf = recv_buf0.Slice(0, vals[0]->shape()[0]);
      }
      CopyFromTo(*vals[0], &recv_buf); // promise thre are no zeros rows
      real_t* data = static_cast<real_t*>(recv_buf.data().dptr_);
      size_t size = recv_buf.shape().Size();
      //std::cout << i << ":vals.size():" << vals.size() << "," << vals[0]->shape() << recv_buf.shape() << ori_shape << (size_t)(&recv_buf) << std::endl;

      auto pull_from_servers = [this, key, data, size, ori_shape, ori_index](
          RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        PSKV& pskv = EncodeKey_Partial(key, ori_shape, ori_index);
      //  std::cout << "pull_pskv.lens:" << pskv.lens.size() << std::endl;
        // issue pull, false means no delete
        auto vals = new ps::SArray<real_t>(data, size, false);
        auto shape2d = ori_shape.FlatTo2D();
        ps::SArray<int> ori_shape0(2);
        ori_shape0[0] = shape2d[0];
        ori_shape0[1] = shape2d[1];
       // CHECK(ori_shape0[1]==128) << ", " << ori_shape0 << std::endl;
        ps::SArray<int> ori_index0;
        ori_index0.CopyFrom(ori_index.data(), ori_index.size());
        CHECK_NOTNULL(ps_worker_)->ZPull_Partial(
            pskv.keys, ori_shape0, ori_index0, pskv.ori_lens,
            vals, pskv.lens, 1, [vals, cb](){ delete vals; cb(); });
      };

      CHECK_NOTNULL(Engine::Get())->PushAsync(
          pull_from_servers,
          pinned_ctx_,
          {},
          {recv_buf.var()},
          FnProperty::kNormal, priority);
      
     // std::cout << recv_buf.shape() 
     //           << ", " << vals[0]->shape()
     //           << ", " << grouped_vals[i].size()
     //           << ", " << grouped_vals[i][0]->shape()
     //           << std::endl;
      comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
     // std::cout << "cc pull partial pass..." << std::endl;
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
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = comm_buf_[key];
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_);
      }
      real_t* data = static_cast<real_t*>(recv_buf.data().dptr_);
      size_t size = recv_buf.shape().Size();

      auto pull_from_servers = [this, key, data, size](
          RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        PSKV& pskv = EncodeKey(key, size);

        // issue pull, false means no delete
        auto vals = new ps::SArray<real_t>(data, size, false);
        CHECK_NOTNULL(ps_worker_)->ZPull(
        pskv.keys, vals, &pskv.lens, 0, [vals, cb](){ delete vals; cb(); });
      };

      CHECK_NOTNULL(Engine::Get())->PushAsync(
          pull_from_servers,
          pinned_ctx_,
          {},
          {recv_buf.var()},
          FnProperty::kNormal, priority);

      comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
    }
  }

  void set_updater(const Updater& updater) override {
    CHECK(updater) << "invalid updater";
    if (IsServerNode()) {
      CHECK_NOTNULL(server_)->set_updater(updater);
    } else {
      updater_ = updater;
    }
  }

  void set_partial_updater(const Partial_Updater& updater, statenum) override {
    CHECK(updater) << "invalid updater";
    if (IsServerNode()) {
      CHECK_NOTNULL(server_)->set_partial_updater(updater, satenum);
    } else {
      partial_satenum = statenum;
      partial_updater_ = updater;
    }
  }

  void Barrier() override {
    ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
  }


  void SendCommandToServers(int cmd_id,
                            const std::string& cmd_body) override {
    CHECK_NOTNULL(ps_worker_);
    ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, ps::kServerGroup));
  }

  int get_group_size() const override { return ps::NumWorkers(); }

  int get_rank() const override { return ps::MyRank(); }

  int get_num_dead_node(int node_id, int timeout) const override {
    int number = 0;
    auto dead_nodes = ps::Postoffice::Get()->GetDeadNodes(timeout);
    const auto& watch_nodes = ps::Postoffice::Get()->GetNodeIDs(node_id);
    std::unordered_set<int> watch_set(watch_nodes.begin(), watch_nodes.end());
    for (int r : dead_nodes) {
      if (watch_set.find(r) != watch_set.end()) number++;
    }
    return number;
  }

  void RunServer(const Controller& controller) override {
    CHECK(!IsWorkerNode());
    if (IsServerNode()) {
      server_ = new KVStoreDistServer();
      server_->set_controller(controller);
    }

    ps::StartAsync("mxnet_server\0");
    if (!ps::Postoffice::Get()->is_recovery()) {
      ps::Postoffice::Get()->Barrier(
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    }
    if (server_) server_->Run();
    ps::Finalize();
    if (server_) {
      delete server_;
    }
    server_ = nullptr;
  }

 private:
  void Push_Partial_(const std::vector<int>& keys,
             const std::vector<NDArray>& values,
             const std::vector<TShape>& ori_shapes,
             const std::vector<Intlist>& ori_indexes,
             int priority,
             bool do_merge)  {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    std::vector<TShape> grouped_ori_shapes;
    std::vector<Intlist> grouped_ori_indexes;
    GroupKVPairs_Partial(keys, values, ori_shapes, ori_indexes,
        &uniq_keys, &grouped_vals, &grouped_ori_shapes, &grouped_ori_indexes);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devcies
      int key = uniq_keys[i];
      const auto& vals = grouped_vals[i];
      const TShape& ori_shape = grouped_ori_shapes[i];
      const Intlist& ori_index = grouped_ori_indexes[i];
      // check index order
      for (size_t idx = 0; idx < ori_index.size() - 1; idx++)
      {
        CHECK_LT(ori_index[idx], ori_index[idx+1]) \
           << "The original indexes and related data must be ascending sorted, " \
           << idx << "[" << ori_index.size() << "]:" << ori_index[idx];
      }
      NDArray merged = do_merge ? comm_->Reduce(key, vals, priority) : vals[0];

      NDArray send_buf;
      auto& send_buf0 = comm_buf_[key];
      if (merged.ctx().dev_mask() == cpu::kDevMask) {
        send_buf0 = merged;  // avoid memory copy
        send_buf = send_buf0;
      } else {
        if (send_buf0.is_none()) {
          send_buf0 = NDArray(merged.shape(), pinned_ctx_);
          send_buf = send_buf0;
        }  else {
          if (merged.shape()[0] > send_buf0.shape()[0]) {
            send_buf0 = NDArray(merged.shape(), pinned_ctx_);
          }
          CHECK(send_buf0.shape()[0] >= merged.shape()[0]) << "Push_Partial:merge Shape_0 must be LE than send_buf0 shape_0.";
          send_buf = send_buf0.Slice(0, merged.shape()[0]);
        }
        CopyFromTo(merged, &send_buf);
      }

      // push to servers
      size_t size = send_buf.shape().Size();
      real_t* data = static_cast<real_t*>(send_buf.data().dptr_);
      auto push_to_servers =
          [this, key, data, size, ori_shape, ori_index](RunContext rctx, Engine::CallbackOnComplete cb) {
         // convert to ps keys
        PSKV& pskv = EncodeKey_Partial(key, ori_shape, ori_index);

        // do push. false means no delete
        ps::SArray<real_t> vals(data, size, false);
        auto shape2d = ori_shape.FlatTo2D();
        ps::SArray<int> ori_shape0((int*)shape2d.shape_, 2, false);
        ps::SArray<int> ori_index0;
        ori_index0.CopyFrom(ori_index.data(), ori_index.size());
        CHECK_NOTNULL(ps_worker_)->ZPush_Partial(
            pskv.keys, vals, pskv.lens, ori_shape0, ori_index0,
            pskv.ori_lens, 1, [cb]() { cb(); });
      };
      Engine::Get()->PushAsync(
          push_to_servers,
          pinned_ctx_,
          {send_buf.var()},
          {},
          FnProperty::kNormal, priority);
    }
  }

  void Push_(const std::vector<int>& keys,
             const std::vector<NDArray>& values,
             int priority,
             bool do_merge)  {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devcies
      int key = uniq_keys[i];
      const auto& vals = grouped_vals[i];
      NDArray merged = do_merge ? comm_->Reduce(key, vals, priority) : vals[0];

      auto& send_buf = comm_buf_[key];
      if (merged.ctx().dev_mask() == cpu::kDevMask) {
        send_buf = merged;  // avoid memory copy
      } else {
        if (send_buf.is_none()) {
          send_buf = NDArray(merged.shape(), pinned_ctx_);
        }
        CopyFromTo(merged, &send_buf);
      }

      // push to servers
      size_t size = send_buf.shape().Size();
      real_t* data = static_cast<real_t*>(send_buf.data().dptr_);
      auto push_to_servers =
          [this, key, data, size](RunContext rctx, Engine::CallbackOnComplete cb) {
         // convert to ps keys
        PSKV& pskv = EncodeKey(key, size);

        // do push. false means no delete
        ps::SArray<real_t> vals(data, size, false);
        CHECK_NOTNULL(ps_worker_)->ZPush(
        pskv.keys, vals, pskv.lens, 0, [cb]() { cb(); });
      };
      Engine::Get()->PushAsync(
          push_to_servers,
          pinned_ctx_,
          {send_buf.var()},
          {},
          FnProperty::kNormal, priority);
    }
  }

  /**
   * \brief check if the keys are all unique
   */
  void CheckUnique(const std::vector<int>& keys) {
    auto keys_copy = keys;
    auto last = std::unique(keys_copy.begin(), keys_copy.end());
    CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
             static_cast<size_t>(keys.size()));
  }

  /**
   * \brief struct for ps keys and lens
   */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    ps::SArray<int> ori_lens;
    int size;
  };

  /**
   * \brief cache all key partitions
   */
  std::unordered_map<int, PSKV> ps_kv_;

  /**
   * \brief serizelize EncodeKey
   */
  std::mutex mu_;

  /**
   * \brief convert to keys in ps
   */
  inline PSKV& EncodeKey(int key, size_t size) {
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();

    if (!pskv.keys.empty()) {
      CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed";
    } else {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      // a simple heuristic for load balance
      if (size < bigarray_bound_) {
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(size);
        pskv.size = size;
      } else {
        // parition it to all servers
        pskv.size = 0;
        for (int i = 0; i < num_servers; ++i) {
          size_t part_size =
              static_cast<size_t>(static_cast<double>(size)/num_servers*(i+1)) -
              static_cast<size_t>(static_cast<double>(size)/num_servers*i);
          ps::Key ps_key = krs[i].begin() + key;
          CHECK_LT(ps_key, krs[i].end());
          pskv.keys.push_back(ps_key);
          pskv.lens.push_back(part_size);
          pskv.size += part_size;
        }
        CHECK_EQ(static_cast<size_t>(pskv.size), size);
      }
    }
    return pskv;
  }

  inline PSKV& EncodeKey_Partial(int key,
                                 const TShape& ori_shape,
                                 const Intlist& ori_index) {
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();

    pskv.size = 0;
    pskv.keys.clear();
    pskv.ori_lens.clear();
    pskv.lens.clear();
    {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      auto shape2d = ori_shape.FlatTo2D();
      size_t rownum = shape2d[0];
      size_t dimnum = shape2d[1];

      size_t size = ori_index.size() * dimnum;

      Intlist::const_iterator oribegin = ori_index.begin();
      Intlist::const_iterator oritail = ori_index.end();
      // push a negative key
      pskv.keys.push_back(0);
      pskv.size = 0;
      // find end of negative row
      int realstart = std::lower_bound(oribegin, oritail, 0) - oribegin;
//      if (realstart > 10) {
//        for (int i = 0; i < 20; i++) std::cout << realstart-10+i << ":" << ori_index[realstart-10+i] << ",";
//        std::cout << std::endl;
//      }
      pskv.lens.push_back(realstart * dimnum);
      pskv.ori_lens.push_back(0);
      pskv.size += realstart * dimnum;

      // a simple heuristic for load balance
      if (false && size < bigarray_bound_) {
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(size - realstart * dimnum);
        pskv.ori_lens.push_back(rownum * dimnum);
        pskv.size += size - realstart * dimnum;
      } else {
        // parition it to all servers
        size_t find_index = 0;
        for (int i = 0; i < num_servers; ++i) {
          size_t part_size =
              static_cast<size_t>(static_cast<double>(rownum)/num_servers*(i+1)) -
              static_cast<size_t>(static_cast<double>(rownum)/num_servers*i);
          CHECK_GT(part_size, 0) << "part_size must be larger than 0";
          pskv.ori_lens.push_back(part_size * dimnum);
          ps::Key ps_key = krs[i].begin() + key;
          CHECK_LT(ps_key, krs[i].end());
          pskv.keys.push_back(ps_key);
          find_index += part_size;
//      if (realstart > 10) {
//        for (int i = 0; i < 20; i++) std::cout << i << "[:]" << ori_index[i] << ",";
//        std::cout << std::endl;
//      }
          int findwhat = find_index-1;
//          std::cout << i << ":" << findwhat << std::endl;
          int realpart_end = std::upper_bound(oribegin, oritail, findwhat) - oribegin;
          int realpart_size = realpart_end - realstart;
//          std::cout << i << ":" << find_index-1 << "," << oritail - oribegin 
//                    << "," << realstart << "," << realpart_end
//                    << "," << realpart_size << std::endl;
          pskv.lens.push_back(realpart_size * dimnum);
          pskv.size += realpart_size * dimnum;
          realstart = realpart_end;
        }
        CHECK_EQ(static_cast<size_t>(pskv.size), size) << ", " << pskv.size << ", " << size;
      }
    }
//    std::cout << "pskv:" << pskv.lens << std::endl;
    return pskv;
  }

  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<real_t>* ps_worker_;
  /**
   * \brief the server handle
   */
  KVStoreDistServer* server_;
  /**
   * \brief threshold for partition
   */
  size_t bigarray_bound_;
  /// \brief send & recver buffer
  std::unordered_map<int, NDArray> comm_buf_;
  std::unordered_map<int, NDArray> comm_buf1_;
};

}  // namespace kvstore
}  // namespace mxnet


#endif  // MXNET_KVSTORE_KVSTORE_DIST_H_
