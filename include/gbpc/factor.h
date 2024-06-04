#ifndef GBPC_FACTOR_H_
#define GBPC_FACTOR_H_

#include <Eigen/Eigen>

#include "gbpc/robust_kernels.h"
#include "gbpc/variable.h"

namespace gbpc {

//! this is unary factor, because the messages will be generated from gtsam
template <int Dim, class GroupOps> class Factor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using RobustKernel = Huber<Dim>;
  using VariableT = Variable<Dim>;
  using Vector = Eigen::Vector<double, Dim>;

  Factor(VariableT::Ptr adj_var, Vector noise_var,
         std::unique_ptr<RobustKernel> robust_kernel = nullptr)
      : adj_var_(adj_var), noise_var_(noise_var),
        robust_kernel_(std::move(robust_kernel)) {}

  void update(Gaussian<Dim> message) {
    if (robust_kernel_) {
      Vector dx = GroupOps::dx(adj_var_->mu(), message.mu_);
      robust_kernel_->filter(&message, dx);
    }

    adj_var_->update({message});
  }

protected:
  VariableT::Ptr adj_var_;

  Vector noise_var_;

  std::unique_ptr<RobustKernel> robust_kernel_;
};

} // namespace gbpc

#endif // GBPC_FACTOR_H_
