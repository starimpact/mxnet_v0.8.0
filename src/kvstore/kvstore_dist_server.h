/*!
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include "ps/ps.h"
#include "mxnet/kvstore.h"
#include "./comm.h"

namespace mxnet {
namespace kvstore {

static const int kStopServer = -1;
static const int kSyncMode = -2;

/**
 * \brief executor runs a function using the thread called \ref Start
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();});
      Block blk = std::move(queue_.front());
      queue_.pop();
      lk.unlock();

      if (blk.f) {
        blk.f(); blk.p->set_value();
      } else {
        blk.p->set_value(); break;
      }
      lk.lock();
    }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief let the thread called \ref Start to exec a function. threadsafe
   */
  void Exec(const Func& func) {
    Block blk(func);
    auto fut = blk.p->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one();
    }
    fut.wait();
  }

  /**
   * \brief stop the thread, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<float>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandle, this, _1, _2, _3));
    ps_server_->set_request_partial_handle(
        std::bind(&KVStoreDistServer::DataHandle_Partial, this, _1, _2, _3));
    sync_mode_ = false;
  }

  ~KVStoreDistServer() {
    delete ps_server_;
  }

  void set_controller(const KVStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  void set_partial_updater(const KVStore::Partial_Updater& updater)  {
    CHECK(updater);
    partial_updater_ = updater;
  }
  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    if (recved.head == kStopServer) {
      exec_.Stop();
    } else if (recved.head == kSyncMode) {
      sync_mode_ = true;
    } else {
      // let the main thread to execute ctrl, which is necessary for python
      exec_.Exec([this, recved]() {
          CHECK(controller_);
          controller_(recved.head, recved.body);
        });
    }
    app->Response(recved);
  }

  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<real_t>& req_data,
                  ps::KVServer<real_t>* server) {
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }

    int key = DecodeKey(req_data.keys[0]);
    auto& stored = store_[key];

    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      size_t ds[] = {(size_t)req_data.lens[0]};
      TShape dshape(ds, ds + 1);
      real_t *pdata = (real_t*)req_data.vals.data();
      if (std::isnan(pdata[0]) || std::isinf(pdata[0])) {
        std::cout << "server DataHandle, encounter a nan or inf:" << pdata[0] << std::endl;
        server->Response(req_meta);
        return;
      }
      TBlob recv_blob(pdata, // NOLINT(*)
                      dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);
      if (stored.is_none()) {
        // initialization
        stored = NDArray(dshape, Context());
        CopyFromTo(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];

        merged.request.push_back(req_meta);

        exec_.Exec([this, key, &recved, &stored](){
            CHECK(updater_);
            updater_(key, recved, &stored);
          });
        stored.WaitToRead();

        if (merged.request.size() == (size_t)ps::NumWorkers()) {
          // let the main thread to execute updater_, which is necessary for
          // python
          for (const auto& req : merged.request) {
            server->Response(req);
          }
          merged.request.clear();
        }
      } else {
        // async push
        exec_.Exec([this, key, &recved, &stored](){
            CHECK(updater_);
            updater_(key, recved, &stored);
          });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {
      // pull
      ps::KVPairs<real_t> response;
      CHECK(!stored.is_none()) << "init " << key << " first";
      int len = stored.shape()[0];
      response.keys = req_data.keys;
      response.lens = {len};
      // TODO(mli) try to remove this CopyFrom
      response.vals.CopyFrom(static_cast<const float*>(stored.data().dptr_), len);
      server->Response(req_meta, response);
    }
  }

  void DataHandle_Partial(const ps::KVMeta& req_meta,
                  const ps::KVPairs_Partial<real_t>& req_data,
                  ps::KVServer<real_t>* server) {
#define DBG_SHOW_TIME 0
    // do some check
   // std::cout << "DataHandle_Partial" << std::endl;
    CHECK_EQ(req_meta.cmd, 1) << "The req_meta.cmd must be 1";
    CHECK_EQ(req_data.keys.size(), (size_t)1) << ", " << req_data.keys.size();
    int ori_row = 0, dim = 0;
    if (req_meta.push) {
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }

    CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.ori_lens.size(), (size_t)1);
    CHECK_EQ(req_data.ori_shape.size(), (size_t)2);

    ori_row = req_data.ori_shape[0];
    dim = req_data.ori_shape[1];
 
    CHECK_EQ(req_data.ori_index.size(), (std::size_t)req_data.lens[0]/dim);
    CHECK_EQ(req_data.ori_lens[0], ori_row * dim);

    int key = DecodeKey(req_data.keys[0]);
    auto& stored = store_[key];
    auto& state = states_[key];
    auto& stored_buf = store_buffer_[key];
    auto& state_buf= states_buffer_[key];
    auto& grad_buf= grad_buffer_[key];

   // std::cout << "server, ori_index:" << req_data.ori_index << std::endl;
    std::vector<int> ori_index(req_data.ori_index.begin(), req_data.ori_index.end());

    TShape rsv_dshape(2), store_dshape(2);

    rsv_dshape[0] = req_data.lens[0] / dim;
    rsv_dshape[1] = dim;
    store_dshape[0] = ori_row;
    store_dshape[1] = dim;

    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      real_t *pdata = (real_t*)req_data.vals.data();
      if (std::isnan(pdata[0]) || std::isinf(pdata[0])) {
        std::cout << "server DataHandle_Partial, encounter a nan or inf:" << pdata[0] << std::endl;
        server->Response(req_meta);
        return;
      }
      TBlob recv_blob(pdata, // NOLINT(*)
                      rsv_dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);
    //  std::cout << "server handle push" << std::endl;

      if (stored.is_none()) {
        // initialization
        // for the initialization, rsv_dshape is actually the original weight shape of this server.
        stored = NDArray(rsv_dshape, Context());
        state = NDArray(rsv_dshape, Context());
        stored_buf = NDArray(rsv_dshape, Context());
        state_buf = NDArray(rsv_dshape, Context());
        grad_buf = NDArray(rsv_dshape, Context());
        CopyFromTo(recved, &stored, 0);
        state = 0.f;
        server->Response_Partial(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];

        merged.request.push_back(req_meta);

        NDArray store_partial = stored_buf.Slice(0, rsv_dshape[0]);
        NDArray state_partial = state_buf.Slice(0, rsv_dshape[0]);
        NDArray grad_partial = grad_buf.Slice(0, rsv_dshape[0]);

        CopyFromTo_IndexFrom(state, &state_partial, ori_index, 0);
        CopyFromTo_IndexFrom(stored, &store_partial, ori_index, 0);
        CopyFromTo(recved, &grad_partial, 0);

        exec_.Exec([this, key, &grad_partial, &ori_index, &store_partial, &state_partial](){
            CHECK(partial_updater_);
            partial_updater_(key, grad_partial, &store_partial, &state_partial);
          });
        store_partial.WaitToRead();
        state_partial.WaitToRead();

        CopyFromTo_IndexTo(state_partial, &state, ori_index, 0);
        CopyFromTo_IndexTo(store_partial, &stored, ori_index, 0);

        if (merged.request.size() == (size_t)ps::NumWorkers()) {
          for (const auto& req : merged.request) {
            server->Response_Partial(req);
          }
          merged.request.clear();
        }
      } else {
        // async push
#if DBG_SHOW_TIME 
        clock_t start, end;
        float cost0, cost1, cost2;

        clock_t start1, end1;
        float cost_0, cost_1;

        start = clock();
#endif
        NDArray store_partial = stored_buf.Slice(0, rsv_dshape[0]);
        NDArray state_partial = state_buf.Slice(0, rsv_dshape[0]);
        NDArray grad_partial = grad_buf.Slice(0, rsv_dshape[0]);

#if DBG_SHOW_TIME 
        start1 = clock();
#endif
        CopyFromTo_IndexFrom(state, &state_partial, ori_index, 0);
#if DBG_SHOW_TIME 
        end1 = clock();
        cost_0 = (float)(end1 - start1)*1000 / CLOCKS_PER_SEC;
#endif

#if DBG_SHOW_TIME 
        start1 = clock();
#endif
        state_partial.WaitToRead();
#if DBG_SHOW_TIME 
        end1 = clock();
        cost_1 = (float)(end1 - start1)*1000 / CLOCKS_PER_SEC;
#endif

        CopyFromTo_IndexFrom(stored, &store_partial, ori_index, 0);
        store_partial.WaitToRead();

        CopyFromTo(recved, &grad_partial, 0);
        grad_partial.WaitToRead();

#if DBG_SHOW_TIME 
        end = clock();
        cost0 = (float)(end - start)*1000 / CLOCKS_PER_SEC;
#endif
        //printf("async push time cost:%.3f ms", cost * 1000);

#if DBG_SHOW_TIME 
        start = clock();
#endif

        exec_.Exec([this, key, &grad_partial, &ori_index, &store_partial, &state_partial](){
            CHECK(partial_updater_);
            partial_updater_(key, grad_partial, &store_partial, &state_partial);
          });
        server->Response_Partial(req_meta);
        store_partial.WaitToRead();
        state_partial.WaitToRead();

#if DBG_SHOW_TIME 
        end = clock();
        cost1 = (float)(end - start)*1000 / CLOCKS_PER_SEC;
#endif
 
#if DBG_SHOW_TIME 
        start = clock();
#endif

        //std::cout << "server push handle:" << ori_index.size() << std::endl;
        CopyFromTo_IndexTo(state_partial, &state, ori_index, 0);
        CopyFromTo_IndexTo(store_partial, &stored, ori_index, 0);
        stored.WaitToRead();
        state.WaitToRead();
       
#if DBG_SHOW_TIME 
        end = clock();
        cost2 = (float)(end - start)*1000 / CLOCKS_PER_SEC;
        //printf("async push time cost:%.3f ms", cost * 1000);
        std::cout << "async push time cost[ms]: " 
                  << cost0 << ", " << cost1 << ", " << cost2 
                  << "; " << cost_0 << ", " << cost_1
                  << "; " << rsv_dshape
                  << std::endl;
#endif
      }
    } else {
      // pull
#if DBG_SHOW_TIME 
      clock_t start, end;
      float cost;
      start = clock();
#endif

      ps::KVPairs_Partial<real_t> response;
      CHECK(!stored.is_none()) << "init " << key << " first";
      NDArray store_partial = stored_buf.Slice(0, rsv_dshape[0]);
      CopyFromTo_IndexFrom(stored, &store_partial, ori_index, 0);
      store_partial.WaitToRead();
      response.keys = req_data.keys;
      response.lens = req_data.lens;
      response.ori_shape = req_data.ori_shape;
      response.ori_lens = req_data.ori_lens;
      response.ori_index = req_data.ori_index;
      //std::cout << "server pull handle:" << req_data.ori_index.size() << std::endl;
      // TODO(mli) try to remove this CopyFrom
      size_t len = req_data.lens[0];
      response.vals.CopyFrom(static_cast<const float*>(store_partial.data().dptr_), len);
     // std::cout << "server handle pull:" << store_partial.shape() 
     //           << ", " << response.vals << std::endl;
#if DBG_SHOW_TIME 
      end = clock();
      cost = (float)(end - start) / CLOCKS_PER_SEC;
      //printf("async puull time cost:%.3f ms", cost * 1000);
      std::cout << "async pull time cost: " << cost * 1000 << " ms." << std::endl;
#endif
      server->Response_Partial(req_meta, response);
    }
  }

  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }

  /**
   * \brief user defined
   */
  bool sync_mode_;
  KVStore::Controller controller_;
  KVStore::Updater updater_;
  KVStore::Partial_Updater partial_updater_;

  std::unordered_map<int, NDArray> store_;
  std::unordered_map<int, NDArray> states_;
  std::unordered_map<int, NDArray> store_buffer_;
  std::unordered_map<int, NDArray> states_buffer_;
  std::unordered_map<int, NDArray> grad_buffer_;


  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    NDArray array;
    NDArray array_tmp;
  };
  std::unordered_map<int, MergeBuf> merge_buf_;

  Executor exec_;

  ps::KVServer<float>* ps_server_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
