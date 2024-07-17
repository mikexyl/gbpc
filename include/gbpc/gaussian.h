#ifndef GBPC_GAUSSIAN_H_
#define GBPC_GAUSSIAN_H_

#include <Eigen/Eigen>

namespace gbpc {

template <int Dim> class Gaussian {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Mu = Eigen::Vector<double, Dim>;
  using Sigma = Eigen::Matrix<double, Dim, Dim>;

  Gaussian(Eigen::Vector<double, Dim> eta = Eigen::Vector<double, Dim>::Zero(),
           Eigen::Matrix<double, Dim, Dim> lambda =
               Eigen::Matrix<double, Dim, Dim>::Zero(),
           size_t N = 0)
      : eta_(eta), lambda_(lambda), N_(N) {
    Sigma_ = lambda.inverse();
    mu_ = Sigma_ * eta;
  }

  Gaussian(Eigen::Vector<double, Dim> eta,
           Eigen::Matrix<double, Dim, Dim> lambda,
           Eigen::Vector<double, Dim> mu, Eigen::Matrix<double, Dim, Dim> Sigma,
           size_t N = 0)
      : mu_(mu), eta_(eta), Sigma_(Sigma), lambda_(lambda), N_(N) {}

  static Gaussian<Dim> fromMuSigma(Eigen::Vector<double, Dim> mu,
                                   Eigen::Matrix<double, Dim, Dim> Sigma,
                                   size_t N = 0) {
    auto lambda = Sigma.inverse();
    auto eta = lambda * mu;
    return Gaussian<Dim>(eta, lambda, mu, Sigma, N);
  }

  size_t N() const { return N_; }
  auto mu() const { return mu_; }
  auto sigma() const { return Sigma_; }

  Eigen::Vector<double, Dim> mu_, eta_;
  Eigen::Matrix<double, Dim, Dim> Sigma_, lambda_;
  size_t N_;
};
} // namespace gbpc

#endif // GBPC_GAUSSIAN_H_
