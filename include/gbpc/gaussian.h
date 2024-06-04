#ifndef GBPC_GAUSSIAN_H_
#define GBPC_GAUSSIAN_H_

#include <Eigen/Eigen>

namespace gbpc {

template <int Dim> class Gaussian {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Gaussian(Eigen::Vector<double, Dim> eta = Eigen::Vector<double, Dim>::Zero(),
           Eigen::Matrix<double, Dim, Dim> lambda =
               Eigen::Matrix<double, Dim, Dim>::Zero())
      : eta_(eta), lambda_(lambda) {
    Sigma_ = lambda.inverse();
    mu_ = Sigma_ * eta;
  }

  Gaussian(Eigen::Vector<double, Dim> eta,
           Eigen::Matrix<double, Dim, Dim> lambda,
           Eigen::Vector<double, Dim> mu, Eigen::Matrix<double, Dim, Dim> Sigma)
      : mu_(mu), eta_(eta), Sigma_(Sigma), lambda_(lambda) {}

  static Gaussian<Dim> fromMuSigma(Eigen::Vector<double, Dim> mu,
                                   Eigen::Matrix<double, Dim, Dim> Sigma) {
    auto lambda = Sigma.inverse();
    auto eta = lambda * mu;
    return Gaussian<Dim>(eta, lambda, mu, Sigma);
  }

  Eigen::Vector<double, Dim> mu_, eta_;
  Eigen::Matrix<double, Dim, Dim> Sigma_, lambda_;
};
} // namespace gbpc

#endif // GBPC_GAUSSIAN_H_
