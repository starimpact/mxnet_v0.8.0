/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_KV_APP_H_
#define PS_KV_APP_H_
#include <algorithm>
#include <utility>
#include <vector>
#include "ps/base.h"
#include "ps/simple_app.h"
namespace ps {

/**
 * \brief the structure for a list of key-value pairs
 *
 * The keys must be unique and sorted in an increasing order.  The length of a
 * value can be more than one. If \a lens is empty, then the length
 * of a value is determined by `k=vals.size()/keys.size()`.  The \a i-th KV pair
 * is then
 *
 * \verbatim {keys[i], (vals[i*k], ..., vals[(i+1)*k-1])} \endverbatim
 *
 * If \a lens is given, then `lens[i]` is the length of the \a i-th
 * value. Let
 *
 * \verbatim n = lens[0] + .. + lens[i-1]  \endverbatim
 *
 * then the \a i-th KV pair is presented as
 *
 * \verbatim {keys[i], (vals[n], ..., vals[lens[i]+n-1])} \endverbatim
 */
template <typename Val>
struct KVPairs {
  // /** \brief empty constructor */
  // KVPairs() {}
  /** \brief the list of keys */
  SArray<Key> keys;
  /** \brief the according values */
  SArray<Val> vals;
  /** \brief the according value lengths (could be empty) */
  SArray<size_t> lens;
};

template <typename Val>
struct KVPairs_Partial {
  // /** \brief empty constructor */
  // KVPairs() {}
  /** \brief the list of keys */
  SArray<Key> keys;
  /** \brief the according values */
  SArray<Val> vals;
  /** \brief the according value lengths (could be empty) */
  SArray<size_t> lens;

  SArray<size_t> ori_lens;

  SArray<int> ori_shape;

  SArray<int> ori_index;
};

/**
 * \brief A worker node that can \ref Push (\ref Pull) key-value pairs to (from) server
 * nodes
 *
 * \tparam Val the type of value, which should be primitive types such as
 * int32_t and float
 */
template<typename Val>
class KVWorker : public SimpleApp {
 public:
  /** avoid too many this-> */
  using SimpleApp::obj_;
  /**
   * \brief callback function for \ref Push and \ref Pull
   *
   * It is called by the data receiving thread of this instance when the push or
   * pull is actually finished. Namely the kv pairs have already written into
   * servers' data structure or the kv pairs have already pulled back.
   */
  using Callback = std::function<void()>;

  /**
   * \brief constructor
   *
   * \param app_id the app id, should match with \ref KVServer's id
   */
  explicit KVWorker(int app_id) : SimpleApp() {
    using namespace std::placeholders;
    slicer_ = std::bind(&KVWorker<Val>::DefaultSlicer, this, _1, _2, _3);
    slicer_partial_ = std::bind(&KVWorker<Val>::DefaultSlicer_Partial, this, _1, _2, _3);
    obj_ = new Customer(app_id, std::bind(&KVWorker<Val>::Process, this, _1));
  }

  /** \brief deconstructor */
  virtual ~KVWorker() { delete obj_; obj_ = nullptr; }

  /**
   * \brief Pushes a list of key-value pairs to all server nodes.
   *
   * This function pushes a KV list specified by \a keys and \a vals to all
   * server nodes.
   *
   * Sample usage: the following codes push two KV pairs `{1, (1.1, 1.2)}` and `{3,
   * (3.1,3.2)}` to server nodes, where the value is a length-2 float vector
   * \code
   *   KVWorker<float> w;
   *   std::vector<Key> keys = {1, 3};
   *   std::vector<float> vals = {1.1, 1.2, 3.1, 3.2};
   *   w.Push(keys, vals);
   * \endcode
   *
   * If \a lens is given, then the value can be various length. See
   * \ref KVPairs for more information.
   *
   * The KV list is partitioned and sent based on the key range each server
   * maintaining. This function returns without waiting the data are sent
   * actually. Instead, use either \ref Wait or the callback to know when
   * finished. This function is thread-safe.
   *
   * @param keys a list of keys, must be unique and sorted in increasing order
   * @param vals the according values
   * @param lens optional, lens[i] stores the value length of the \a
   * i-th KV pair
   * @param cmd an optional command sent to the servers
   * @param cb the callback which is called when the push is finished.
   * @return the timestamp of this request
   */
  int Push(const std::vector<Key>& keys,
           const std::vector<Val>& vals,
           const std::vector<size_t>& lens = {},
           int cmd = 0,
           const Callback& cb = nullptr) {
    return ZPush(
        SArray<Key>(keys), SArray<Val>(vals), SArray<size_t>(lens), cmd, cb);
  }

  int Push_Partial(const std::vector<Key>& keys,
           const std::vector<Val>& vals,
           const SArray<int>& ori_shape,
           const SArray<int>& ori_index,
           const SArray<size_t>& ori_lens,
           const std::vector<size_t>& lens,
           int cmd = 1,
           const Callback& cb = nullptr) {

    return ZPush_Partial(
        SArray<Key>(keys), SArray<Val>(vals), ori_shape,
        ori_index, ori_lens, SArray<size_t>(lens), cmd, cb);
  }

  /**
   * \brief Pulls the values associated with the keys from the server nodes
   *
   * This function pulls the values of the keys specified in \a keys from the
   * server nodes. The format is same to \ref KVPairs
   *
   * Sample usage: the following codes pull the values of keys \a 1 and \a 3
   * from the server nodes.
   * \code
   *   KVWorker<float> w;
   *   std::vector<Key> keys = {1, 3};
   *   std::vector<float> vals;
   *   ps.Pull(keys, &vals);
   * \endcode
   *
   * It's a non-blocking call. The actual pulling is finished,
   * namely \a vals (and \a lens) is filled with pulled values, only
   * if \ref Wait returns or the callback is called.
   *
   * @param keys a list of keys, must be unique and sorted in increasing order
   * @param vals the buffer for the pulled values. It can be 0 size.
   * @param lens optional buffer for the value length. If set, it can be 0 size.
   * @param cmd an optional command sent to the servers
   * @param cb the callback which is called when the pull is finished.
   * @return the timestamp of this request
   */
  int Pull(const std::vector<Key>& keys,
           std::vector<Val>* vals,
           std::vector<size_t>* lens = nullptr,
           int cmd = 0,
           const Callback& cb = nullptr) {
    return Pull_(SArray<Key>(keys), vals, lens, cmd, cb);
  }

  /**
   * \brief Waits until a push or pull has been finished
   *
   * Sample usage:
   * \code
   *   int ts = w.Pull(keys, &vals);
   *   Wait(ts);
   *   // now vals is ready for use
   * \endcode
   *
   * \param timestamp the timestamp returned by the push or pull
   */
  void Wait(int timestamp) { obj_->WaitRequest(timestamp); }

  /**
   * \brief zero-copy Push
   *
   * This function is similar to \ref Push except that all data
   * will not be copied into system for better performance. It is the caller's
   * responsibility to keep the content to be not changed before actually
   * finished.
   */
  int ZPush(const SArray<Key>& keys,
            const SArray<Val>& vals,
            const SArray<size_t>& lens = {},
            int cmd = 0,
            const Callback& cb = nullptr) {
    int ts = obj_->NewRequest(kServerGroup);
    AddCallback(ts, cb);
    KVPairs<Val> kvs;
    kvs.keys = keys;
    kvs.vals = vals;
    kvs.lens = lens;
    Send(ts, true, cmd, kvs);
    return ts;
  }

  int ZPush_Partial(const SArray<Key>& keys,
            const SArray<Val>& vals,
            const SArray<size_t>& lens,
            const SArray<int>& ori_shape,
            const SArray<int>& ori_index,
            const SArray<size_t>& ori_lens,
            int cmd = 1,
            const Callback& cb = nullptr) {
    int ts = obj_->NewRequest(kServerGroup);
    AddCallback(ts, cb);
    KVPairs_Partial<Val> kvs;
    kvs.keys = keys;
    kvs.vals = vals;
    kvs.lens = lens;
    kvs.ori_lens = ori_lens;
    kvs.ori_shape = ori_shape;
    kvs.ori_index = ori_index;
    Send_Partial(ts, true, cmd, kvs);
    return ts;
  }

  /**
   * \brief zero-copy Pull
   *
   * This function is similar to \ref Pull except that all data
   * will not be copied into system for better performance. It is the caller's
   * responsibility to keep the content to be not changed before actually
   * finished.
   */
  int ZPull(const SArray<Key>& keys,
            SArray<Val>* vals,
            SArray<size_t>* lens = nullptr,
            int cmd = 0,
            const Callback& cb = nullptr) {
    return Pull_(keys, vals, lens, cmd, cb);
  }

  int ZPull_Partial(const SArray<Key>& keys,
            const SArray<int>& ori_shape,
            const SArray<int>& ori_index,
            const SArray<size_t>& ori_lens,
            SArray<Val>* vals,
            const SArray<size_t>& lens,
            int cmd = 0,
            const Callback& cb = nullptr) {
    return Pull_Partial_(keys, ori_shape, ori_index, ori_lens, vals, lens, cmd, cb);
  }

  using SlicedKVs = std::vector<std::pair<bool, KVPairs<Val>>>;

  using SlicedKVs_Partial = std::vector<std::pair<bool, KVPairs_Partial<Val>>>;
  /**
   * \brief a slicer partitions a key-value list according to the key ranges
   * \param send the kv list for partitioning
   * \param ranges the key ranges, ranges[i] is the key range of server i
   * \param sliced the sliced lists. slices[i] should only contains keys in
   * ranges[i] and the according values
   */
  using Slicer = std::function<void(
      const KVPairs<Val>& send, const std::vector<Range>& ranges,
      SlicedKVs* sliced)>;

  using Slicer_Partial = std::function<void(
      const KVPairs_Partial<Val>& send, const std::vector<Range>& ranges,
      SlicedKVs_Partial* sliced)>;
  /**
   * \brief set a user-defined slicer
   */
  void set_slicer(const Slicer& slicer) {
    CHECK(slicer); slicer_ = slicer;
  }

  void set_slicer_partial(const Slicer& slicer) {
    CHECK(slicer); slicer_partial_ = slicer;
  }
 private:
  /**
   * \brief internal pull, C/D can be either SArray or std::vector
   */
  template <typename C, typename D>
  int Pull_(const SArray<Key>& keys, C* vals, D* lens,
            int cmd, const Callback& cb);

  template <typename C>
  int Pull_Partial_(
    const SArray<Key>& keys, const SArray<int>& ori_shape,
    const SArray<int>& ori_index, const SArray<size_t>& ori_lens,
    C* vals, const SArray<size_t>& lens, int cmd, const Callback& cb);
  /**
   * \brief add a callback for a request. threadsafe.
   * @param cb callback
   * @param timestamp the timestamp of the request
   */
  void AddCallback(int timestamp, const Callback& cb) {
    if (!cb) return;
    std::lock_guard<std::mutex> lk(mu_);
    callbacks_[timestamp] = cb;
  }

  /**
   * \brief run and delete the callback
   * \param timestamp the timestamp of the callback
   */
  void RunCallback(int timestamp);
  /**
   * \brief send the kv list to all servers
   * @param timestamp the timestamp of the request
   * @param push whether or not it is a push request
   * @param cmd command
   */
  void Send(int timestamp, bool push, int cmd, const KVPairs<Val>& kvs);

  void Send_Partial(int timestamp, bool push, int cmd, const KVPairs_Partial<Val>& kvs);
  /** \brief internal receive handle */
  void Process(const Message& msg);
  /** \brief default kv slicer */
  void DefaultSlicer(const KVPairs<Val>& send,
                     const std::vector<Range>& ranges,
                     SlicedKVs* sliced);

  void DefaultSlicer_Partial(const KVPairs_Partial<Val>& send,
                     const std::vector<Range>& ranges,
                     SlicedKVs_Partial* sliced);

  /** \brief data buffer for received kvs for each timestamp */
  std::unordered_map<int, std::vector<KVPairs<Val>>> recv_kvs_;

  std::unordered_map<int, std::vector<KVPairs_Partial<Val>>> recv_kvs_partial_;
  /** \brief callbacks for each timestamp */
  std::unordered_map<int, Callback> callbacks_;
  /** \brief lock */
  std::mutex mu_;
  /** \brief kv list slicer */
  Slicer slicer_;

  Slicer_Partial slicer_partial_;
};

/** \brief meta information about a kv request */
struct KVMeta {
  /** \brief the int cmd */
  int cmd;
  /** \brief whether or not this is a push request */
  bool push;
  /** \brief sender's node id */
  int sender;
  /** \brief the associated timestamp */
  int timestamp;
};

/**
 * \brief A server node for maintaining key-value pairs
 */
template <typename Val>
class KVServer : public SimpleApp {
 public:
  /**
   * \brief constructor
   * \param app_id the app id, should match with \ref KVWorker's id
   */
  explicit KVServer(int app_id) : SimpleApp() {
    using namespace std::placeholders;
    obj_ = new Customer(app_id, std::bind(&KVServer<Val>::Process, this, _1));
  }

  /** \brief deconstructor */
  virtual ~KVServer() { delete obj_; obj_ = nullptr; }

  /**
   * \brief the handle to process a push/pull request from a worker
   * \param req_meta meta-info of this request
   * \param req_data kv pairs of this request
   * \param server this pointer
   */
  using ReqHandle = std::function<void(const KVMeta& req_meta,
                                       const KVPairs<Val>& req_data,
                                       KVServer* server)>;
  using Req_PartialHandle = std::function<void(const KVMeta& req_meta,
                                       const KVPairs_Partial<Val>& req_data,
                                       KVServer* server)>;

  void set_request_handle(const ReqHandle& request_handle) {
    CHECK(request_handle) << "invalid request handle";
    request_handle_ = request_handle;
  }

  void set_request_partial_handle(const Req_PartialHandle& request_handle) {
    CHECK(request_handle) << "invalid request partial handle";
    request_partial_handle_ = request_handle;
  }
  /**
   * \brief response to the push/pull request
   * \param req the meta-info of the request
   * \param res the kv pairs that will send back to the worker
   */
  void Response(const KVMeta& req, const KVPairs<Val>& res = KVPairs<Val>());

  void Response_Partial(const KVMeta& req, const KVPairs_Partial<Val>& res = KVPairs_Partial<Val>());

 private:
  /** \brief internal receive handle */
  void Process(const Message& msg);
  /** \brief request handle */
  ReqHandle request_handle_;
  Req_PartialHandle request_partial_handle_;
};


/**
 * \brief an example handle adding pushed kv into store
 */
template <typename Val>
struct KVServerDefaultHandle {
  void operator()(
      const KVMeta& req_meta, const KVPairs<Val>& req_data, KVServer<Val>* server) {
    size_t n = req_data.keys.size();
    KVPairs<Val> res;
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
    } else {
      res.keys = req_data.keys; res.vals.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
      Key key = req_data.keys[i];
      if (req_meta.push) {
        store[key] += req_data.vals[i];
      } else {
        res.vals[i] = store[key];
      }
    }
    server->Response(req_meta, res);
  }
  std::unordered_map<Key, Val> store;
};


///////////////////////////////////////////////////////////////////////////////

template <typename Val>
void KVServer<Val>::Process(const Message& msg) {
  if (msg.meta.simple_app) {
    SimpleApp::Process(msg); return;
  }
  KVMeta meta;
  meta.cmd       = msg.meta.head;
  meta.push      = msg.meta.push;
  meta.sender    = msg.meta.sender;
  meta.timestamp = msg.meta.timestamp;

  int n = msg.data.size();
  if (meta.cmd != 1)
  {
    CHECK_EQ(meta.cmd, 0) << "the meta.cmd must be 0 when doing normal push";
    KVPairs<Val> data;
    if (n) {
      CHECK_GE(n, 2);
      data.keys = msg.data[0];
      data.vals = msg.data[1];
      if (n > 2) {
        CHECK_EQ(n, 3);
        data.lens = msg.data[2];
        CHECK_EQ(data.lens.size(), data.keys.size());
      }
    }
    CHECK(request_handle_);
    request_handle_(meta, data, this);
  } else if (meta.cmd == 1) {
  //  std::cout << "server process, cmd:" << meta.cmd << std::endl;
    KVPairs_Partial<Val> data;
    CHECK_EQ(n, 6) << "The partial info size must be 6";
    data.keys = msg.data[0];
    data.vals = msg.data[1];
    data.lens = msg.data[2];
    data.ori_lens = msg.data[3];
    data.ori_shape = msg.data[4];
    data.ori_index = msg.data[5];
    CHECK(request_partial_handle_);
  //  std::cout << "server process, before request_partial_handle." << std::endl;
    request_partial_handle_(meta, data, this);
  }
}

template <typename Val>
void KVServer<Val>::Response(const KVMeta& req, const KVPairs<Val>& res) {
  Message msg;
  msg.meta.customer_id = obj_->id();
  msg.meta.request     = false;
  msg.meta.push        = req.push;
  msg.meta.head        = req.cmd;
  msg.meta.timestamp   = req.timestamp;
  msg.meta.recver      = req.sender;
  if (res.keys.size()) {
    msg.AddData(res.keys);
    msg.AddData(res.vals);
    if (res.lens.size()) {
      msg.AddData(res.lens);
    }
  }
  Postoffice::Get()->van()->Send(msg);
}

template <typename Val>
void KVServer<Val>::Response_Partial(const KVMeta& req, const KVPairs_Partial<Val>& res) {
  Message msg;
  msg.meta.customer_id = obj_->id();
  msg.meta.request     = false;
  msg.meta.push        = req.push;
  msg.meta.head        = req.cmd;
  msg.meta.timestamp   = req.timestamp;
  msg.meta.recver      = req.sender;
  if (res.keys.size()) {
    msg.AddData(res.keys);
    msg.AddData(res.vals);
    msg.AddData(res.lens);
    msg.AddData(res.ori_lens);
    msg.AddData(res.ori_shape);
    msg.AddData(res.ori_index);
  }
  Postoffice::Get()->van()->Send(msg);
}

template <typename Val>
void KVWorker<Val>::DefaultSlicer(
    const KVPairs<Val>& send, const std::vector<Range>& ranges,
    typename KVWorker<Val>::SlicedKVs* sliced) {
  sliced->resize(ranges.size());

  // find the positions in msg.key
  size_t n = ranges.size();
  std::vector<size_t> pos(n+1);
  const Key* begin = send.keys.begin();
  const Key* end = send.keys.end();
  for (size_t i = 0; i < n; ++i) {
    if (i == 0) {
      pos[0] = std::lower_bound(begin, end, ranges[0].begin()) - begin;
      begin += pos[0];
    } else {
      CHECK_EQ(ranges[i-1].end(), ranges[i].begin());
    }
    size_t len = std::lower_bound(begin, end, ranges[i].end()) - begin;
    begin += len;
    pos[i+1] = pos[i] + len;

    // don't send it to severs for empty kv
    sliced->at(i).first = (len != 0);
  }
  CHECK_EQ(pos[n], send.keys.size());
  if (send.keys.empty()) return;

  // the length of value
  size_t k = 0, val_begin = 0, val_end = 0;
  if (send.lens.empty()) {
    k = send.vals.size() / send.keys.size();
    CHECK_EQ(k * send.keys.size(), send.vals.size());
  } else {
    CHECK_EQ(send.keys.size(), send.lens.size());
  }

  // slice
  for (size_t i = 0; i < n; ++i) {
    if (pos[i+1] == pos[i]) {
      sliced->at(i).first = false;
      continue;
    }
    sliced->at(i).first = true;
    auto& kv = sliced->at(i).second;
    kv.keys = send.keys.segment(pos[i], pos[i+1]);
    if (send.lens.size()) {
      kv.lens = send.lens.segment(pos[i], pos[i+1]);
      for (size_t l : kv.lens) val_end += l;
      kv.vals = send.vals.segment(val_begin, val_end);
      val_begin = val_end;
    } else {
      kv.vals = send.vals.segment(pos[i]*k, pos[i+1]*k);
    }
  }
}

template <typename Val>
void KVWorker<Val>::DefaultSlicer_Partial(
    const KVPairs_Partial<Val>& send, const std::vector<Range>& ranges,
    typename KVWorker<Val>::SlicedKVs_Partial* sliced) {
  sliced->resize(ranges.size());

  // find the positions in msg.key
  size_t n = ranges.size();
  std::vector<size_t> pos(n+1);

  const Key* begin = send.keys.begin();
  const Key* end = send.keys.end();
  for (size_t i = 0; i < n; ++i) {
    if (i == 0) {
      // +1 is to skip the first key.
      pos[0] = std::lower_bound(begin + 1, end, ranges[0].begin()) - begin;
      begin += pos[0];
    } else {
      CHECK_EQ(ranges[i-1].end(), ranges[i].begin());
    }
    size_t len = std::lower_bound(begin, end, ranges[i].end()) - begin;
    begin += len;
    pos[i+1] = pos[i] + len;
    // don't send it to severs for empty kv
    sliced->at(i).first = (len != 0);
  }
  CHECK_EQ(pos[n], send.keys.size());
  if (send.keys.size() <= 1) return;

  // the length of value
  size_t val_begin = 0, val_end = 0, oval_begin = 0, oval_end = 0;
  const SArray<int>& ori_shape = send.ori_shape;
  const SArray<int>& ori_index = send.ori_index;
  size_t dim = ori_shape[1];
  size_t ori_size = ori_index.size() * dim;
  if (!send.lens.empty()) ori_size -= send.lens[0];

  CHECK_GT(send.lens.size(), 1) << "The send.lens.size() must be larger than 1";

  CHECK_EQ(send.keys.size(), send.lens.size()) << "The key size must equal lens size";

  CHECK_EQ(send.ori_lens.size(), send.lens.size()) << "The ori_lens size must equal lens size";

 // std::cout << "default_slice_partial, send.lens:" << send.lens << std::endl;
  // slice
  val_begin = send.lens[0];
  val_end = val_begin;
  for (size_t i = 0; i < n; ++i) {
  //  std::cout << "default_slice_partial, pos:" << pos[i] << "," << pos[i+1] << std::endl;
    if (pos[i+1] == pos[i]) {
      sliced->at(i).first = false;
      continue;
    }
    sliced->at(i).first = true;
    auto& kv = sliced->at(i).second;
//    CHECK(pos[i] <= pos[i+1]) << ", -----1\n";
    kv.keys = send.keys.segment(pos[i], pos[i+1]);
    kv.ori_shape.resize(2);
    kv.ori_shape[1] = dim;
    
//    CHECK(pos[i] <= pos[i+1]) << ", -----2\n";
    kv.lens = send.lens.segment(pos[i], pos[i+1]);
    for (size_t l : kv.lens) val_end += l;
    if (send.vals.size() > 0) {
//      CHECK(val_begin <= val_end) << ", -----3\n";
      kv.vals = send.vals.segment(val_begin, val_end);
    }
    size_t row_begin = val_begin / dim;
    size_t row_end = val_end / dim;
//    CHECK(row_begin <= row_end) << ", " << send.vals.size()
//            << ", " << i << ", " << kv.lens << ", " << send.lens
//            << ", " << row_begin << ", " << row_end
//            << ", -----4\n";
    kv.ori_index = send.ori_index.segment(row_begin, row_end);

   // std::cout << "default_slice_partial, " << i 
   //           << ", kv.ori_index:" << kv.ori_index << std::endl;

//    CHECK(pos[i] <= pos[i+1]) << ", -----5\n";
    kv.ori_lens = send.ori_lens.segment(pos[i], pos[i+1]);
    for (size_t l : kv.ori_lens) oval_end += l;
    size_t orow_begin = oval_begin / dim;
    size_t orow_end = oval_end / dim;
    kv.ori_shape[0] = orow_end - orow_begin;

    for (int& idx : kv.ori_index) idx -= orow_begin;

    val_begin = val_end;
    oval_begin = oval_end;

//    std::cout << "default_slice_partial, " << i 
//              << ", kv.ori_shape:" << kv.ori_shape << std::endl;
//
//    std::cout << "default_slice_partial, " << i 
//              << ", kv.lens:" << kv.lens << std::endl;
//
//    std::cout << "default_slice_partial, " << i 
//              << ", kv.vals:" << kv.vals << std::endl;
//
//    std::cout << "default_slice_partial, " << i 
//              << ", kv.ori_lens:" << kv.ori_lens << std::endl;


  }
}

template <typename Val>
void KVWorker<Val>::Send(int timestamp, bool push, int cmd, const KVPairs<Val>& kvs) {
  // slice the message
  SlicedKVs sliced;
  slicer_(kvs, Postoffice::Get()->GetServerKeyRanges(), &sliced);

  // need to add response first, since it will not always trigger the callback
  int skipped = 0;
  for (size_t i = 0; i < sliced.size(); ++i) {
    if (!sliced[i].first) ++skipped;
  }
  obj_->AddResponse(timestamp, skipped);
  if ((size_t)skipped == sliced.size()) {
    RunCallback(timestamp);
  }

  for (size_t i = 0; i < sliced.size(); ++i) {
    const auto& s = sliced[i];
    if (!s.first) continue;
    Message msg;
    msg.meta.customer_id = obj_->id();
    msg.meta.request     = true;
    msg.meta.push        = push;
    msg.meta.head        = cmd;
    msg.meta.timestamp   = timestamp;
    msg.meta.recver      = Postoffice::Get()->ServerRankToID(i);
    const auto& kvs = s.second;
    if (kvs.keys.size()) {
      msg.AddData(kvs.keys);
      msg.AddData(kvs.vals);
      if (kvs.lens.size()) {
        msg.AddData(kvs.lens);
      }
    }
    Postoffice::Get()->van()->Send(msg);
  }
}


template <typename Val>
void KVWorker<Val>::Send_Partial(int timestamp, bool push, int cmd, const KVPairs_Partial<Val>& kvs) {
  // slice the message
  SlicedKVs_Partial sliced;
  slicer_partial_(kvs, Postoffice::Get()->GetServerKeyRanges(), &sliced);
  //CHECK(kvs.ori_shape[1]==128) << kvs.ori_shape << std::endl;

 // std::cout << "Send_Partial:after slicer_partial_ push:" << push << std::endl;
  // need to add response first, since it will not always trigger the callback
  int skipped = 0;
  for (size_t i = 0; i < sliced.size(); ++i) {
    if (!sliced[i].first) ++skipped;
  }
  obj_->AddResponse(timestamp, skipped);
  if ((size_t)skipped == sliced.size()) {
    RunCallback(timestamp);
  }

  for (size_t i = 0; i < sliced.size(); ++i) {
    const auto& s = sliced[i];
    if (!s.first) continue;
    Message msg;
    msg.meta.customer_id = obj_->id();
    msg.meta.request     = true;
    msg.meta.push        = push;
    msg.meta.head        = cmd;
    msg.meta.timestamp   = timestamp;
    msg.meta.recver      = Postoffice::Get()->ServerRankToID(i);
    const auto& kvs = s.second;
    if (kvs.keys.size()) {
      msg.AddData(kvs.keys);
      msg.AddData(kvs.vals);
      msg.AddData(kvs.lens);
      msg.AddData(kvs.ori_lens);
      msg.AddData(kvs.ori_shape);
      msg.AddData(kvs.ori_index);
    }
   // std::cout << "ready to send " << i << "'th message. push:" << push << std::endl;
    Postoffice::Get()->van()->Send(msg);
  }
  //CHECK(kvs.ori_shape[1]==128) << kvs.ori_shape << std::endl;
}

template <typename Val>
void KVWorker<Val>::Process(const Message& msg) {
  if (msg.meta.simple_app) {
    SimpleApp::Process(msg); return;
  }

  // store the data for pulling
  int ts = msg.meta.timestamp;
  if (!msg.meta.push && msg.data.size()) {
    CHECK_GE(msg.data.size(), (size_t)2);
    if (msg.meta.head == 0) {
      KVPairs<Val> kvs;
      kvs.keys = msg.data[0];
      kvs.vals = msg.data[1];
      if (msg.data.size() > (size_t)2) {
        kvs.lens = msg.data[2];
      }
      mu_.lock();
      recv_kvs_[ts].push_back(kvs);
      mu_.unlock();
    } else if (msg.meta.head == 1) {
      KVPairs_Partial<Val> kvs;
      kvs.keys = msg.data[0];
      kvs.vals = msg.data[1];
      kvs.lens = msg.data[2];
      kvs.ori_lens = msg.data[3];
      kvs.ori_shape = msg.data[4];
      kvs.ori_index = msg.data[5];

      mu_.lock();
      recv_kvs_partial_[ts].push_back(kvs);
      mu_.unlock();
    }
  }

  // finished, run callbacks
  if (obj_->NumResponse(ts) == Postoffice::Get()->num_servers() - 1)  {
    RunCallback(ts);
  }
}

template <typename Val>
void KVWorker<Val>::RunCallback(int timestamp) {
  mu_.lock();
  auto it = callbacks_.find(timestamp);
  if (it != callbacks_.end()) {
    mu_.unlock();

    CHECK(it->second);
    it->second();

    mu_.lock();
    callbacks_.erase(it);
  }
  mu_.unlock();
}

template <typename Val>
template <typename C, typename D>
int KVWorker<Val>::Pull_(
    const SArray<Key>& keys, C* vals, D* lens, int cmd, const Callback& cb) {
  int ts = obj_->NewRequest(kServerGroup);
  AddCallback(ts, [this, ts, keys, vals, lens, cb]() mutable {
      mu_.lock();
      auto& kvs = recv_kvs_[ts];
      mu_.unlock();

      // do check
      size_t total_key = 0, total_val = 0;
      for (const auto& s : kvs) {
        Range range = FindRange(keys, s.keys.front(), s.keys.back()+1);
        CHECK_EQ(range.size(), s.keys.size())
            << "unmatched keys size from one server";
        if (lens) CHECK_EQ(s.lens.size(), s.keys.size());
        total_key += s.keys.size();
        total_val += s.vals.size();
      }
      CHECK_EQ(total_key, keys.size()) << "lost some servers?";

      // fill vals and lens
      std::sort(kvs.begin(), kvs.end(), [](
          const KVPairs<Val>& a, const KVPairs<Val>& b) {
                  return a.keys.front() < b.keys.front();
        });
      CHECK_NOTNULL(vals);
      if (vals->empty()) {
        vals->resize(total_val);
      } else {
        CHECK_EQ(vals->size(), total_val);
      }
      Val* p_vals = vals->data();
      size_t *p_lens = nullptr;
      if (lens) {
        if (lens->empty()) {
          lens->resize(keys.size());
        } else {
          CHECK_EQ(lens->size(), keys.size());
        }
        p_lens = lens->data();
      }
      for (const auto& s : kvs) {
        memcpy(p_vals, s.vals.data(), s.vals.size() * sizeof(Val));
        p_vals += s.vals.size();
        if (p_lens) {
          memcpy(p_lens, s.lens.data(), s.lens.size() * sizeof(size_t));
          p_lens += s.lens.size();
        }
      }

      mu_.lock();
      recv_kvs_.erase(ts);
      mu_.unlock();
      if (cb) cb();
    });

  KVPairs<Val> kvs; kvs.keys = keys;
  Send(ts, false, cmd, kvs);
  return ts;
}

template <typename Val>
template <typename C>
int KVWorker<Val>::Pull_Partial_(
    const SArray<Key>& keys, const SArray<int>& ori_shape,
    const SArray<int>& ori_index, const SArray<size_t>& ori_lens,
    C* vals, const SArray<size_t>& lens, int cmd, const Callback& cb) {
  int ts = obj_->NewRequest(kServerGroup);
  //CHECK(ori_shape[1]==128) << "," << ori_shape << std::endl;
  AddCallback(ts, [this, ts, keys, ori_shape, ori_index, 
                   ori_lens, vals, lens, cb]() mutable {
      mu_.lock();
      auto& kvs = recv_kvs_partial_[ts];
      mu_.unlock();
      //CHECK(ori_shape[1]==128) << "," << "Pull_Partial_ Callback" << ori_shape << ori_index << ori_lens << *vals << lens << std::endl;

      // do check
      size_t total_key = 0, total_val = 0;
      for (const auto& s : kvs) {
        Range range = FindRange(keys, s.keys.front(), s.keys.back()+1);
        CHECK_EQ(range.size(), s.keys.size())
            << "unmatched keys size from one server";
        CHECK_EQ(s.lens.size(), s.keys.size());
        total_key += s.keys.size();
        total_val += s.vals.size();
      }
      CHECK_EQ(total_key + 1, keys.size()) << "keys: lost some servers?";

      int dim = ori_shape[1];
      const int* oribegin = ori_index.begin();
      const int* oritail = ori_index.end();
      // find end of negative row
      int realstart = std::lower_bound(oribegin, oritail, 0) - oribegin;
      realstart *= dim;
      total_val += realstart;

      // fill vals and lens
      std::sort(kvs.begin(), kvs.end(), [](
          const KVPairs_Partial<Val>& a, const KVPairs_Partial<Val>& b) {
                  return a.keys.front() < b.keys.front();
        });

      CHECK_EQ(vals->size(), total_val) 
          << ", " << vals->size() << ", " 
          << total_val << ", " << realstart << ", "
          << ori_shape << ", " << lens << ", "
          << total_key << ", " << ori_index << std::endl;

      Val* p_vals = vals->data();
      CHECK_EQ(lens.size(), keys.size());
      
      p_vals += realstart;
      size_t idx = 1;
      size_t srcsize = 0;
      for (const auto& s : kvs) {
        memcpy(p_vals, s.vals.data(), s.vals.size() * sizeof(Val));
        srcsize += s.vals.size();
        p_vals += s.vals.size();
        CHECK_EQ(lens[idx], s.lens[0])
            << ", " << idx << ":" << lens[idx]
            << ", " << s.lens[0] << std::endl;
        CHECK_EQ(s.vals.size(), s.lens[0])
            << ", " << idx << ":" << lens[idx]
            << ", " << s.vals << std::endl;
        idx++;
      }
      CHECK_EQ(p_vals, vals->end()) << p_vals << "," << vals->end() << std::endl;

      mu_.lock();
      recv_kvs_partial_.erase(ts);
      mu_.unlock();
      if (cb) cb();
    });

 // std::cout << "Pull_Partial_" << std::endl;
  KVPairs_Partial<Val> kvs;
  kvs.keys = keys;
  kvs.lens = lens;
  kvs.ori_shape = ori_shape;
  kvs.ori_index = ori_index;
  kvs.ori_lens = ori_lens;
  Send_Partial(ts, false, cmd, kvs);
  return ts;
}

}  // namespace ps
#endif  // PS_KV_APP_H_
