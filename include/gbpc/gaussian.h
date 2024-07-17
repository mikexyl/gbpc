#ifndef GBPC_GAUSSIAN_H_
#define GBPC_GAUSSIAN_H_

#include <Eigen/Eigen>
#include <optional>

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

  double hellingerDistance(const Gaussian<Dim> &other) const {
    return hellingerDistance(mu_, other.mu_, Sigma_, other.Sigma_);
  }

  static double hellingerDistance(const Eigen::VectorXd &mu1,
                                  const Eigen::VectorXd &mu2,
                                  const Eigen::MatrixXd &cov1,
                                  const Eigen::MatrixXd &cov2) {
    // Ensure dimensions match
    assert(mu1.size() == mu2.size() &&
           "Mean vectors must have the same dimension");
    assert(cov1.rows() == cov1.cols() && "Covariance matrix 1 must be square");
    assert(cov2.rows() == cov2.cols() && "Covariance matrix 2 must be square");
    assert(cov1.rows() == cov2.rows() &&
           "Covariance matrices must have the same dimensions");

    // Compute the combined covariance matrix
    Eigen::MatrixXd covAvg = 0.5 * (cov1 + cov2);

    // Compute the determinant and inverse of the covariance matrices
    double detCov1 = cov1.determinant();
    double detCov2 = cov2.determinant();
    double detCovAvg = covAvg.determinant();

    // Compute the normalization coefficient
    double coeff = std::sqrt(std::sqrt(detCov1 * detCov2) / detCovAvg);

    // Compute the Mahalanobis distance term
    Eigen::VectorXd diff = mu1 - mu2;
    double mahalanobis = diff.transpose() * covAvg.inverse() * diff;
    double exponent = -0.125 * mahalanobis;

    // Compute the Hellinger distance
    double hellinger = std::sqrt(1.0 - coeff * std::exp(exponent));

    return hellinger;
  }

  void updateMoments() {
    Sigma_ = lambda_.inverse();
    mu_ = Sigma_ * eta_;
  }

  void updateCanonical() {
    lambda_ = Sigma_.inverse();
    eta_ = lambda_ * mu_;
  }

  void relax(double k) {
    lambda_ *= k;
    eta_ *= k;

    updateMoments();
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
