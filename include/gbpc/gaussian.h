#ifndef GBPC_GAUSSIAN_H_
#define GBPC_GAUSSIAN_H_

#include <Eigen/Eigen>

namespace gbpc {

template <int Dim> struct Gaussian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Gaussian(Eigen::Vector<double, Dim> eta = Eigen::Vector<double, Dim>::Zero(),
           Eigen::Matrix<double, Dim, Dim> lambda =
               Eigen::Matrix<double, Dim, Dim>::Zero())
      : eta(eta), lambda(lambda) {}

  static Gaussian<Dim> fromMuSigma(Eigen::Vector<double, Dim> mu,
                                   Eigen::Matrix<double, Dim, Dim> Sigma) {
    auto lambda = Sigma.inverse();
    auto eta = lambda * mu;
    return Gaussian<Dim>(eta, lambda);
  }

  Eigen::Vector<double, Dim> eta;
  Eigen::Matrix<double, Dim, Dim> lambda;
};
} // namespace gbpc

#endif // GBPC_GAUSSIAN_H_
