/*!
 * Copyright (c) 2017 by MingZhang
 * \file take_op-inl.h
 * \brief take element from a tensor.
*/
#ifndef MXNET_OPERATOR_TAKE_INL_H_
#define MXNET_OPERATOR_TAKE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace take_ {
enum TakeOpInputs {kData, kIndex};
enum TakeOpOutputs {kOut};
}  //namespace take_

struct TakeParam : public dmlc::Parameter<TakeParam> {
  
};

template<typename xpu>
class TakeOp : public Operator {
 public:
  explicit TakeOp(TakeParam p) : param_(p) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &qux_args) {
    using namespace mshadow;
    
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
  }

 private:
  TakeParam param_;
};  // class TakeOp

template<typename xpu>
Operator* CreateOp(TakeParam param);


#if DMLC_USE_CXX11
class TakeProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "index"};
  }
  
  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParam() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
  }

  OperatorProperty* Copy() const override {
    TakeProp* sym = new TakeProp();
    sym->param_ = this->param_;
  }

  std::string TypeString() const override {
    return "Take";
  }

  std::vector<int> DeclareBackwardDependencya(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[take_::kOut], in_data[take_::kIndex]};
  }

  Operator* CreateOperator(Context ctx) const override;
 private:
  TakeParam param_;
};  // class TakeProp
#endif

}  // namespace op
}  // namespace mxnet

#endif
