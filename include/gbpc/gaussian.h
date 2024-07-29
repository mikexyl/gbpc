#ifndef GBPC_GAUSSIAN_H_
#define GBPC_GAUSSIAN_H_

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <Eigen/Eigen>
#include <concepts>
#include <optional>

namespace gtsam {
class NoiseModelValue {};
}  // namespace gtsam

using namespace gtsam;

template <typename T>
concept PoseConcept = std::same_as<T, Pose3> || std::same_as<T, Pose2> ||
                      std::same_as<T, Point2> || std::same_as<T, Point3>;

namespace gbpc {

enum class GaussianMergeType { Merge, MergeRobust, Mixture, Replace };

class Gaussian {
 public:
  using shared_ptr = std::shared_ptr<Gaussian>;
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using This = Gaussian;

  Gaussian() = default;
  Gaussian(const Gaussian& other) = default;
  Gaussian(Gaussian&& other) = default;
  Gaussian(Key key,
           const Vector& mu,
           const Vector& eta,
           const Eigen::MatrixXd& Sigma,
           const Eigen::MatrixXd& lambda,
           size_t N)
      : mu_(mu), eta_(eta), Sigma_(Sigma), lambda_(lambda), N_(N), key_(key) {}
  Gaussian(Key key, const Vector& mu, const Eigen::MatrixXd& Sigma, size_t N)
      : mu_(mu), Sigma_(Sigma), N_(N), key_(key) {
    updateCanonical();
  }

  static auto ToMoments(const Vector& eta, const Eigen::MatrixXd& Lambda) {
    auto Sigma = Lambda.inverse();
    auto mu = Sigma * eta;
    return std::make_pair(mu, Sigma);
  }

  size_t N() const { return N_; }
  const Vector& mu() const { return mu_; }
  const Vector& eta() const { return eta_; }
  Matrix Sigma() const { return Sigma_; }
  Matrix Lambda() const { return lambda_; }
  Key key() const { return key_; }

  Gaussian& operator=(const Gaussian& other) {
    mu_ = other.mu_;
    eta_ = other.eta_;
    Sigma_ = other.Sigma_;
    lambda_ = other.lambda_;
    N_ = other.N_;

    return *this;
  }

  double hellingerDistance(const This& other) const {
    return hellingerDistance(mu_, other.mu_, Sigma_, other.Sigma_);
  }

  static double hellingerDistance(const Eigen::VectorXd& mu1,
                                  const Eigen::VectorXd& mu2,
                                  const Eigen::MatrixXd& cov1,
                                  const Eigen::MatrixXd& cov2) {
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

  static Gaussian mixtureGaussian(const Gaussian& gauss1,
                                  const Gaussian& gauss2,
                                  std::optional<double> force_alpha = 0.5) {
    assert(gauss1.key() == gauss2.key());
    uint64_t key = gauss1.key();

    double alpha;
    if (force_alpha.has_value()) {
      alpha = force_alpha.value();
    } else {
      if (gauss1.N() == 0 or gauss2.N() == 0) {
        throw std::runtime_error("N is zero");
      }
      alpha = static_cast<double>(gauss1.N()) / (gauss1.N() + gauss2.N());
    }

    auto const &mu1 = gauss1.mu(), mu2 = gauss2.mu();
    Vector mu_mix = alpha * mu1 + (1 - alpha) * mu2;
    Matrix mu1mu1t = mu1 * mu1.transpose();
    Matrix mu2mu2t = mu2 * mu2.transpose();
    Matrix mu_mixmu_mixt = mu_mix * mu_mix.transpose();
    Matrix Sigma_mix = alpha * (gauss1.Sigma() + mu1mu1t) +
                       (1 - alpha) * (gauss2.Sigma() + mu2mu2t) - mu_mixmu_mixt;
    return Gaussian(key, mu_mix, Sigma_mix, (gauss1.N() + gauss2.N()) / 2);
  }

  void merge(const Gaussian& other) {
    assert(key_ == other.key_);
    lambda_ += other.lambda_;
    eta_ += other.eta_;
    N_ = (N_ + other.N_) / 2;

    updateMoments();
  }

  void update(const std::vector<Gaussian>& messages,
              GaussianMergeType merge_type) {
    switch (merge_type) {
      case GaussianMergeType::Merge:
      case GaussianMergeType::MergeRobust: {
        for (auto message : messages) {
          if (merge_type == GaussianMergeType::MergeRobust) {
            double hellinger = this->hellingerDistance(message);
            message.relax(1 - hellinger);
          }
          this->merge(message);
        }

      } break;
      case GaussianMergeType::Mixture: {
        for (const auto& message : messages) {
          // TODO: ! this is not on manifold
          *this = (mixtureGaussian(*this, message));
        }
      } break;
      case GaussianMergeType::Replace: {
        auto message = messages.front();
        *this = message;
      } break;
      default:
        throw std::runtime_error("Unknown GaussianMergeType");
        break;
    }
  }

 protected:
  Eigen::VectorXd mu_, eta_;
  Eigen::MatrixXd Sigma_, lambda_;
  size_t N_;
  Key key_;
};

template <PoseConcept VALUE>
class Belief : public Gaussian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using This = Belief<VALUE>;

  using Mu = VALUE;
  using TangentVector = Vector;
  using Matrix = Eigen::MatrixXd;
  using Covariance = Matrix;
  using Noise = noiseModel::Gaussian;

  Belief() = default;

  Belief(const This& other) = default;
  Belief(const Gaussian& other) : Gaussian(other) {}

  Belief(Key key, const Vector& mu, const Covariance& Sigma, size_t N)
      : Gaussian(key, mu, Sigma, N) {}
};

}  // namespace gbpc

#endif  // GBPC_GAUSSIAN_H_
