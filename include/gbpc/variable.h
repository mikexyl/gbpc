#ifndef GBPC_VARIABLE_NODE_H_
#define GBPC_VARIABLE_NODE_H_

#include <Eigen/Eigen>
#include <functional>
#include <memory>

#include "gaussian.h"

namespace gbpc {

template <int Dim> class Variable {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<Variable<Dim>>;

  Variable(Gaussian<Dim> initial)
      : belief_(initial), mu_(initial.mu_), Sigma_(initial.Sigma_) {}

  void update(const std::vector<Gaussian<Dim>> &messages) {
    auto eta = belief_.eta_;
    auto lambda = belief_.lambda_;

    for (const auto &message : messages) {
      lambda += message.lambda_;
      eta += message.eta_;
    }

    belief_.eta_ = eta;
    belief_.lambda_ = lambda;
    Sigma_ = lambda.inverse();
    mu_ = Sigma_ * belief_.eta_;
  }

  auto mu() const { return mu_; }
  auto sigma() const { return Sigma_; }

protected:
  Gaussian<Dim> belief_;

  Eigen::Vector<double, Dim> mu_;
  Eigen::Matrix<double, Dim, Dim> Sigma_;

  std::function<Eigen::Vector<double, Dim>(Eigen::Vector<double, Dim>)>
      robust_kernel_;
};

} // namespace gbpc

#endif // GBPC_VARIABLE_NODE_H_
