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

namespace gbpc {

enum class GaussianMergeType { Merge, MergeRobust, Mixture, Replace };

class Gaussian {
 public:
  using shared_ptr = std::shared_ptr<Gaussian>;
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using This = Gaussian;

  Gaussian() : N_(0) {}
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
  Key& key() { return key_; }

  Gaussian& operator=(const Gaussian& other) {
    key_ = other.key_;
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
                                  std::optional<double> force_alpha = 0.5,
                                  bool expect_same_key = true) {
    if (expect_same_key) {
      assert(gauss1.key() == gauss2.key());
    } else {
      assert(gauss1.key() != gauss2.key());
    }
    uint64_t key = gauss1.key();

    double alpha;
    if (force_alpha.has_value()) {
      alpha = force_alpha.value();
    } else {
      if (gauss1.N() == 0 and gauss2.N() == 0) {
        throw std::runtime_error("N is zero");
      } else if (gauss1.N() == 0) {
        return gauss2;
      } else if (gauss2.N() == 0) {
        return gauss1;
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

  void add(const Gaussian& other, bool expect_same_key = true) {
    if (N_ == 0) {
      return this->replace(other);
    }

    if (expect_same_key) {
      assert(key_ == other.key_);
    }

    Sigma_ += other.Sigma_;
    mu_ += other.mu_;
    N_ = std::min(N_, other.N_);
    N_++;

    updateCanonical();
  }

  void merge(const Gaussian& other, bool expect_same_key = true) {
    if (N_ == 0) {
      return this->replace(other);
    }

    if (expect_same_key) {
      assert(key_ == other.key_);
    }

    // check size of the matrices
    assert(mu_.size() == other.mu_.size());
    assert(Sigma_.size() == other.Sigma_.size());
    assert(lambda_.size() == other.lambda_.size());
    assert(eta_.size() == other.eta_.size());

    lambda_ += other.lambda_;
    eta_ += other.eta_;
    N_ = std::min(N_, other.N_);
    N_++;

    updateMoments();
  }

  bool equalSize(const Gaussian& other) const {
    return mu_.size() == other.mu_.size() &&
           Sigma_.size() == other.Sigma_.size() &&
           lambda_.size() == other.lambda_.size() &&
           eta_.size() == other.eta_.size();
  }

  void replace(const Gaussian& other) { *this = other; }

  void update(const std::vector<Gaussian>& messages,
              GaussianMergeType merge_type) {
    switch (merge_type) {
      case GaussianMergeType::Merge:
      case GaussianMergeType::MergeRobust: {
        for (auto message : messages) {
          if (merge_type == GaussianMergeType::MergeRobust) {
            double hellinger = this->hellingerDistance(message);
            double k = std::max(0.01, 1 - hellinger);
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
    ss << "key: " << DefaultKeyFormatter(key_) << std::endl;
    ss << "mu: " << mu_.transpose() << std::endl;
    ss << "Sigma: " << Sigma_ << std::endl;
    ss << "N: " << N_ << std::endl;
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const This& obj) {
    os << obj.print();
    return os;
  }

  bool empty() const { return mu_.size() == 0; }

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

  void send(const shared_ptr& exclude = nullptr) {
    for (auto neighbor : neighbors_) {
      if (neighbor != exclude) sendToNeighbor(neighbor);
    }
  }

  void sendToNeighbor(const shared_ptr& receiver) {
    Gaussian message(this->prior());

    std::cout << " send from " << key_ << " to " << receiver->key()
              << std::endl;

    for (auto it = neighbors_.begin(); it != neighbors_.end(); it++) {
      if (*it != receiver) {
        if (messages_.find(*it) != messages_.end()) {
          auto potential = this->potential(*it, messages_[*it]);
          if (not potential.has_value()) {
            message.merge(messages_[*it], false);
          } else {
            message.merge(*potential, false);
          }
          std::cout << "message: " << messages_[*it].mu().transpose()
                    << std::endl;
        } else {
          return;
        }
      }
    }

    if (message.empty()) {
      return;
    }

    message.key() = receiver->key();
    receiver->receive(shared_from_this(), message);
  }

  void receive(const shared_ptr& sender, const Gaussian& message) {
    assert(message.key() == this->key_);
    messages_[sender] = message;
    assert(messages_[sender].key() == this->key_);
  }

  auto const& messages() const { return messages_; }

  virtual Gaussian prior() const { return Gaussian(); }

  virtual std::optional<Gaussian> potential(const Node::shared_ptr& sender,
                                            const Gaussian& message) = 0;

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

  virtual void update() {
    // update belief
    std::cout << "-------" << key() << "--------" << std::endl;
    std::cout << "start :" << mu().transpose() << ","
              << Sigma().diagonal().transpose() << std::endl;
    for (auto const& [_, message] : messages_) {
      assert(message.key() == this->key_);
      std::cout << "message :" << message.mu().transpose() << ","
                << message.Sigma().diagonal().transpose() << std::endl;
      this->Gaussian::update({message}, GaussianMergeType::MergeRobust);
      std::cout << "update :" << mu().transpose() << ","
                << Sigma().diagonal().transpose() << std::endl;
    }
  }

 protected:
  std::map<shared_ptr, Gaussian> messages_;
  std::vector<shared_ptr> neighbors_;
};

template <class VALUE>
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

  virtual std::optional<Gaussian> potential(const Node::shared_ptr& sender,
                                            const Gaussian& message) override {
    return std::nullopt;
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
