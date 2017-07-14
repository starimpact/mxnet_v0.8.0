
#include "./take_op-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp(TakeParam param) {
  return new TakeOp<cpu>(param);
}

Operator* TakeProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(TakeParam);

MXNET_REGISTER_OP_PROPERTY(Take, TakeProp)
.describe("Take indexed element from data.")
.add_argument("data", "Symbol", "Input data.")
.add_argument("index", "Symbol", "Indexed position")
.add_arguments(TakeParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
