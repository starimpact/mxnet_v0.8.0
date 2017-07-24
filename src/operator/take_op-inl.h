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
enum TakeOpResource {kTempSpace};
}  //namespace take_

struct TakeParam : public dmlc::Parameter<TakeParam> {
  DMLC_DECLARE_PARAMETER(TakeParam) {}
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
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    Tensor<xpu, 2> data = in_data[take_::kData].get<xpu, 2, real_t>(s); 
    Tensor<xpu, 1> index = in_data[take_::kIndex].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> out = out_data[take_::kOut].get<xpu, 1, real_t>(s);
//    Tensor<cpu, 1> index_cpu(Shape1(1));AllocSpace(&index_cpu, false);
    Tensor<cpu, 1> index_cpu = ctx.requested[take_::kTempSpace].get_space<cpu>(Shape1(1), s_cpu);
    Copy<1, real_t>(index_cpu, index);
    int idx = static_cast<int>(index_cpu[0]);
    Copy<1, real_t>(out, data[idx], s);
//    FreeSpace(&index_cpu);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    Tensor<xpu, 1> index = in_data[take_::kIndex].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> grad_out = out_grad[take_::kOut].get<xpu, 1, real_t>(s);
    Tensor<xpu, 2> grad_in = in_grad[take_::kData].get<xpu, 2, real_t>(s);
//    Tensor<cpu, 1> index_cpu(Shape1(1));AllocSpace(&index_cpu, false);
    Tensor<cpu, 1> index_cpu = ctx.requested[take_::kTempSpace].get_space<cpu>(Shape1(1), s_cpu);
    Copy<1, real_t>(index_cpu, index);
    if (req[take_::kOut] == kWriteTo) {
      grad_in = 0.f;
    }
    int idx = static_cast<int>(index_cpu[0]);
    grad_in[idx] += grad_out;
//    FreeSpace(&index_cpu);
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

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Take operator must have 2 inputs.";
    const TShape &dshape = (*in_shape)[take_::kData];
    const TShape &ishape = (*in_shape)[take_::kIndex];
    CHECK_EQ(ishape.ndim(), 1) << "index shape must be 1 dimension.";
    CHECK_EQ(ishape[0], 1) << "index must be a scalar.";
    TShape oshape(dshape.data()+1, dshape.data()+dshape.ndim());
    out_shape->clear();
    out_shape->push_back(oshape);

    return true;
  }

  OperatorProperty* Copy() const override {
    TakeProp* sym = new TakeProp();
    sym->param_ = this->param_;
    return sym;
  }

  std::string TypeString() const override {
    return "Take";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[take_::kOut], in_data[take_::kIndex]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override;
 private:
  TakeParam param_;
};  // class TakeProp
#endif

}  // namespace op
}  // namespace mxnet

#endif
