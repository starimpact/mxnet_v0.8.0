/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore.cc
 * \brief implement kv_store
 */
#include <mxnet/kvstore.h>
#include <stdlib.h>
#include <dmlc/logging.h>
#include "./kvstore_local.h"
// #include "./kvstore_device.h"
#if MXNET_USE_DIST_KVSTORE
#include "./kvstore_dist.h"
#endif  // MXNET_USE_DIST_KVSTORE

namespace mxnet {

KVStore* KVStore::Create(const char *type_name) {
  std::string tname = type_name;
  std::transform(tname.begin(), tname.end(), tname.begin(), ::tolower);
  KVStore* kv = nullptr;
  bool use_device_comm = false;
  auto has = [tname](const std::string& pattern) {
    return tname.find(pattern) != std::string::npos;
  };
  if (has("device")) {
    use_device_comm = true;
  }

  if (has("dist")) {
#if MXNET_USE_DIST_KVSTORE
    kv = new kvstore::KVStoreDist(use_device_comm);
    if (!has("_async") && kv->IsWorkerNode() && kv->get_rank() == 0) {
      // configure the server to be the sync mode
      kv->SendCommandToServers(kvstore::kSyncMode, "");
    }
#else
    LOG(FATAL) << "compile with USE_DIST_KVSTORE=1 to use " << tname;
    return nullptr;
#endif  // MXNET_USE_DIST_KVSTORE
  } else {
    kv =  new kvstore::KVStoreLocal(use_device_comm);
  }
  kv->type_ = tname;
  return kv;
}

// copy ndfrom to the indexto positions of ndto.
void CopyFromTo_IndexTo(NDArray& ndfrom, NDArray *ndto, vector<int>& indexto, int priority) {
  TShape& shapefrom = ndfrom.shape();
  CHECK_EQ(shapefrom[0], indexto.size());
  for (int idx = 0; idx < shapefrom[0]; idx++) {
    int idxto = indexto[idx];
    if (idxto < 0) continue;
    CopyFromTo(ndfrom.At(idx), &ndto->At(idxto), priority);
  }
}

// copy data of indexfrom positions of ndfrom to ndto.
void CopyFromTo_IndexFrom(NDArray& ndfrom, NDArray *ndto, vector<int>& indexfrom, int priority) {
  TShape& shapeto = ndto->shape();
  CHECK_EQ(shapeto[0], indexto.size());
  for (int idx = 0; idx < shapeto[0]; idx++) {
    int idxfrom = indexfrom[idx];
    if (idxfrom < 0) continue;
    CopyFromTo(ndfrom.At(idxfrom), &ndto->At(idx), priority);
  }
}

}  // namespace mxnet
