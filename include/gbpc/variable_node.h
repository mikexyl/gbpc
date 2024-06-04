#ifndef GBPC_VARIABLE_NODE_H_
#define GBPC_VARIABLE_NODE_H_

#include <Eigen/Eigen>

#include "gaussian.h"

namespace gbpc {

template <int Dim> class VariableNode {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VariableNode() {}

  void update(const std::vector<Gaussian<Dim>> &messages) {
    auto eta = prior_.eta;
    auto lambda = prior_.lambda;

    for (const auto &message : messages) {
      lambda += message.lambda;
      eta += message.eta;
    }

    belief_.eta = eta;
    belief_.lambda = lambda;
    Sigma_ = lambda.inverse();
    mu_ = Sigma_ * belief_.eta;
  }

  auto mu() const { return mu_; }
  auto sigma() const { return Sigma_; }

protected:
  Gaussian<Dim> prior_;
  Gaussian<Dim> belief_;

  Eigen::Vector<double, Dim> mu_;
  Eigen::Matrix<double, Dim, Dim> Sigma_;
};

} // namespace gbpc

#endif // GBPC_VARIABLE_NODE_H_
