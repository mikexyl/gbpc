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

  using Ptr = std::shared_ptr<Factor<Dim, GroupOps>>;

  using RobustKernel = Huber<Dim>;
  using VariableT = Variable<Dim>;
  using Vector = Eigen::Vector<double, Dim>;

  Factor(VariableT::Ptr adj_var,
         std::unique_ptr<RobustKernel> robust_kernel = nullptr)
      : adj_var_(adj_var), robust_kernel_(std::move(robust_kernel)) {}

  std::string update(Gaussian<Dim> message) {
    std::stringstream ss;
    ss << "Factor::update: " << adj_var_->mu().transpose() << " -> "
       << message.mu_.transpose();

    if (robust_kernel_) {
      Vector dx = GroupOps::dx(adj_var_->mu(), message.mu_);
      robust_kernel_->filter(&message, dx);
    }

    adj_var_->update({message});

    ss << " -> " << adj_var_->mu().transpose();

    return ss.str();
  }

  auto adj_var() { return adj_var_; }

protected:
  VariableT::Ptr adj_var_;

  std::unique_ptr<RobustKernel> robust_kernel_;
};

} // namespace gbpc

#endif // GBPC_FACTOR_H_
