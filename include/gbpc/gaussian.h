#ifndef GBPC_GAUSSIAN_H_
#define GBPC_GAUSSIAN_H_

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

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
  explicit Gaussian(Key key) : key_(key) {}
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

    size_t N1 = gauss1.N(), N2 = gauss2.N();
    size_t weighted_N = (N1 * N1 + N2 * N2) / (N1 + N2);

    return Gaussian(key, mu_mix, Sigma_mix, weighted_N);
  }

  void merge(const Gaussian& other) {
    assert(key_ == other.key_);
    lambda_ += other.lambda_;
    eta_ += other.eta_;
    N_ = (N_ + other.N_) / 2;

    updateMoments();
  }

  void replace(const Gaussian& other) {
    assert(key_ == other.key_);
    mu_ = other.mu_;
    Sigma_ = other.Sigma_;
    N_ = other.N_;
  }

  void update(const std::vector<Gaussian>& messages,
              GaussianMergeType merge_type) {
    switch (merge_type) {
      case GaussianMergeType::Merge:
      case GaussianMergeType::MergeRobust: {
        for (auto message : messages) {
          if (merge_type == GaussianMergeType::MergeRobust) {
            double hellinger = this->hellingerDistance(message);
            double k = std::max(0.1, 1 - hellinger);
            message.relax(k);
          }
          this->merge(message);
        }

      } break;
      case GaussianMergeType::Mixture: {
        for (const auto& message : messages) {
          // TODO: ! this is not on manifold
          this->replace(mixtureGaussian(*this, message));
        }
      } break;
      case GaussianMergeType::Replace: {
        auto message = messages.front();
        this->replace(message);
      } break;
      default:
        throw std::runtime_error("Unknown GaussianMergeType");
        break;
    }
  }

  std::string print() const {
    std::stringstream ss;
    ss << "key: " << key_ << std::endl;
    ss << "mu: " << mu_.transpose() << std::endl;
    ss << "Sigma: " << Sigma_ << std::endl;
    ss << "N: " << N_ << std::endl;
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const This& obj) {
    os << obj.print();
    return os;
  }

 protected:
  Eigen::VectorXd mu_, eta_;
  Eigen::MatrixXd Sigma_, lambda_;
  size_t N_;
  Key key_;
};

class Node : public std::enable_shared_from_this<Node>, public Gaussian {
 public:
  using shared_ptr = std::shared_ptr<Node>;
  Node() = default;
  Node(const Key& key) : Gaussian(key) {}
  Node(const Gaussian& initial) : Gaussian(initial) {}
  virtual ~Node() = default;

  void send(const shared_ptr& receiver) {
    assert(messages_.size() > 0);

    std::optional<Gaussian> message = priorMessage();

    for (auto it = messages_.begin(); it != messages_.end(); it++) {
      if (it->first != receiver) {
        if (not message.has_value()) {
          message = it->second;
        } else {
          message->merge(it->second);
        }
      }
    }

    receiver->receive(shared_from_this(), message.value());
  }

  void receive(const shared_ptr& sender, const Gaussian& message) {
    messages_[sender] = message;
  }

  auto const& messages() const { return messages_; }

  virtual std::optional<Gaussian> priorMessage() const = 0;

  void clearMessages() { messages_.clear(); }

  void addNeighbor(const shared_ptr& neighbor) {
    neighbors_.emplace_back(neighbor);
  }

  auto const& neighbors() const { return neighbors_; }
  void addNeighbors(const std::vector<shared_ptr>& neighbors) {
    for (auto neighbor : neighbors) {
      addNeighbor(neighbor);
    }
  }

 protected:
  std::map<shared_ptr, Gaussian> messages_;
  std::vector<shared_ptr> neighbors_;
};

template <PoseConcept VALUE>
class Belief : public Node {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using This = Belief<VALUE>;

  using Mu = VALUE;
  using TangentVector = Vector;
  using Matrix = Eigen::MatrixXd;
  using Covariance = Matrix;
  using Noise = noiseModel::Gaussian;

  Belief(const Key& key) : Node(key) {}

  Belief(const This& other) = default;
  Belief(const Gaussian& other) : Node(other) {}

  Belief(Key key, const Vector& mu, const Covariance& Sigma, size_t N)
      : Node(Gaussian(key, mu, Sigma, N)) {}

  std::optional<Gaussian> priorMessage() const override {
    if (N_ == 0) {
      return std::nullopt;
    }
    return static_cast<Gaussian>(*this);
  }

  static Gaussian optimizeWithGtsam(const std::vector<This>& beliefs) {
    NonlinearFactorGraph graph;
    Values values;
    for (auto belief : beliefs) {
      auto noise = noiseModel::Gaussian::Covariance(belief.Sigma());
      VALUE value = belief.mu();
      auto factor = NonlinearFactor::shared_ptr(
          new gtsam::PriorFactor<VALUE>(belief.key(), value, noise));
      graph.add(factor);
      values.insert_or_assign(belief.key(), value);
    }

    auto key = beliefs.front().key();

    LevenbergMarquardtOptimizer optimizer(graph, values);
    auto result = optimizer.optimize();

    // get variance from the result
    Marginals marginals(graph, result);
    auto cov = marginals.marginalCovariance(key);

    auto mu = result.at<VALUE>(key);

    return Gaussian(key, mu, cov, 0);
  }
};

}  // namespace gbpc

#endif  // GBPC_GAUSSIAN_H_
