#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <Eigen/Eigen>
#include <optional>

namespace gtsam {
class NoiseModelValue {};
}  // namespace gtsam

using namespace gtsam;

namespace gbpc {

enum class GaussianMergeType {
  Merge = 0,
  MergeRobust,
  Damp,
  Replace,
  Step,
  Contract,
  DampContract
};

enum class MetricType { Hellinger = 0, KLDivergence, Chi2 };

struct UpdateParams {
  GaussianMergeType type = GaussianMergeType::Damp;
  double relax = 1.0;
  bool use_fixed_alpha = false;
  double fixed_alpha = 0;
  float belief_change_threshold = 0.0;
  double step_size = 0.001;
  float gamma = 0.1;
  float d_reset = 0.1;
  float contract_alpha = 0.95;
  MetricType metric_type = MetricType::Hellinger;
};

struct UpdateResult {
  enum Status { Success, Failed };
  std::vector<Status> status;
  std::vector<double> change;
  std::string message;
};

class Gaussian {
 public:
  using shared_ptr = std::shared_ptr<Gaussian>;
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using This = Gaussian;

  Gaussian() = delete;
  Gaussian(Key key,
           const Vector& mu,
           const Vector& eta,
           const Eigen::MatrixXd& Sigma,  // covariance
           const Eigen::MatrixXd& lambda,
           size_t degree)
      : mu_(mu),
        eta_(eta),
        Sigma_(Sigma),
        lambda_(lambda),
        degree_(degree),
        key_(key) {}
  Gaussian(Key key,
           const Vector& mu,
           const Eigen::MatrixXd& Sigma,
           size_t degree)
      : mu_(mu), Sigma_(Sigma), degree_(degree), key_(key) {
    updateCanonical();
  }

  static Gaussian Random(Key key, size_t dim, size_t degree = 1) {
    Vector mu = Vector::Random(dim);
    Matrix Sigma = Matrix::Random(dim, dim) * 100;
    return Gaussian(key, mu, Sigma, degree);
  }

  static auto ToMoments(const Vector& eta, const Eigen::MatrixXd& Lambda) {
    auto Sigma = Lambda.inverse();
    auto mu = Sigma * eta;
    return std::make_pair(mu, Sigma);
  }

  size_t degree() const { return degree_; }
  const Vector& mu() const { return mu_; }
  Vector& mu() { return mu_; }
  const Vector& eta() const { return eta_; }
  Matrix Sigma() const { return Sigma_; }
  Matrix& Sigma() { return Sigma_; }
  Matrix Lambda() const { return lambda_; }
  Key key() const { return key_; }
  Key& key() { return key_; }

  Gaussian operator-(const This& other) const {
    auto epsilon_sigma = Sigma();
    epsilon_sigma.setIdentity();
    epsilon_sigma *= 1e-6;
    return Gaussian(key(),
                    mu() - other.mu(),
                    Sigma() - other.Sigma() + epsilon_sigma,
                    degree());
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

  void shiftMu(const Vector& shift) {
    mu_ += shift;
    updateCanonical();
  }

  static Gaussian Damp(const Gaussian& gauss1,
                       const Gaussian& gauss2,
                       std::optional<double> force_alpha = 0.5,
                       bool expect_same_key = true) {
    if (expect_same_key and gauss1.key() != gauss2.key()) {
      throw std::invalid_argument(
          fmt::format("Keys do not match: {} vs {}",
                      DefaultKeyFormatter(gauss1.key()),
                      DefaultKeyFormatter(gauss2.key())));
    }
    uint64_t key = gauss1.key();

    double alpha;
    if (force_alpha.has_value()) {
      alpha = force_alpha.value();
    } else {
      if (gauss1.degree() == 0 and gauss2.degree() == 0) {
        return gauss2;
      } else if (gauss1.degree() == 0) {
        return gauss2;
      } else if (gauss2.degree() == 0) {
        return gauss1;
      }
      alpha = static_cast<double>(gauss1.degree()) /
              (gauss1.degree() + gauss2.degree());
    }

    auto const &mu1 = gauss1.mu(), mu2 = gauss2.mu();
    Vector mu_mix = alpha * mu1 + (1 - alpha) * mu2;
    Matrix mu1mu1t = mu1 * mu1.transpose() * 2;
    Matrix mu2mu2t = mu2 * mu2.transpose() * 2;
    Matrix mu_mixmu_mixt = mu_mix * mu_mix.transpose();
    Matrix Sigma_mix =
        alpha * (gauss1.Sigma()) + (1 - alpha) * (gauss2.Sigma());
    // Sigma_mix *= 0.7;

    size_t degree1 = gauss1.degree(), degree2 = gauss2.degree();
    size_t weighted_degree =
        (degree1 * degree1 + degree2 * degree2) /
        (degree1 + degree2 + 1);  // + 1 to avoid division by zero

    return Gaussian(key, mu_mix, Sigma_mix, weighted_degree);
  }

  void damp(const Gaussian& other, double alpha = 0.5) {
    // check size of the matrices
    assert(mu_.size() == other.mu_.size());
    assert(Sigma_.size() == other.Sigma_.size());
    assert(lambda_.size() == other.lambda_.size());
    assert(eta_.size() == other.eta_.size());

    this->replace(Gaussian::Damp(*this, other, alpha, false));
  }

  void merge(const Gaussian& other, bool expect_same_key = true) {
    if (degree_ == 0) {
      return this->replace(other);
    }

    if (expect_same_key) {
      assert(key_ == other.key_);
    } else {
      assert(key_ != other.key_);
    }

    // check size of the matrices
    assert(mu_.size() == other.mu_.size());
    assert(Sigma_.size() == other.Sigma_.size());
    assert(lambda_.size() == other.lambda_.size());
    assert(eta_.size() == other.eta_.size());

    lambda_ += other.lambda_;
    eta_ += other.eta_;
    degree_ = std::min(degree_, other.degree_);
    degree_++;

    updateMoments();
  }

  bool equalSize(const Gaussian& other) const {
    return mu_.size() == other.mu_.size() &&
           Sigma_.size() == other.Sigma_.size() &&
           lambda_.size() == other.lambda_.size() &&
           eta_.size() == other.eta_.size();
  }

  void replace(const Gaussian& other) { *this = other; }

  std::string print() const {
    std::stringstream ss;
    ss << "key: " << key_ << std::endl;
    ss << "mu: " << mu_.transpose() << std::endl;
    ss << "Sigma: " << Sigma_ << std::endl;
    ss << "N: " << degree_ << std::endl;
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const This& obj) {
    os << obj.print();
    return os;
  }

  bool empty() const { return mu_.size() == 0; }

  /**
   * @brief update mu, return chi2 if provided
   *
   * @param mu
   * @param chi2
   */
  void updateMu(const Vector& mu) {
    mu_ = mu;
    updateCanonical();
  }

  enum class Status { Converging, Reset, Converged };
  float contractionRate() const { return contraction_rate_; }
  float& contractionRate() { return contraction_rate_; }
  float contractionStepSize() const { return contraction_step_size_; }
  float& contractionStepSize() { return contraction_step_size_; }
  float dxycurr() const { return d_xy_curr_; }
  float& dxycurr() { return d_xy_curr_; }
  float dxy() const { return d_xy_; }
  float& dxy() { return d_xy_; }

 protected:
  Eigen::VectorXd mu_, eta_;
  Eigen::MatrixXd Sigma_, lambda_;
  size_t degree_;
  Key key_;

  float contraction_rate_{1.0};
  float contraction_step_size_{0.0};
  // float contraction_alpha_{0.0};
  float d_xy_{1.};
  float d_xy_curr_{1.};
};

class Node : public std::enable_shared_from_this<Node>, public Gaussian {
 public:
  using shared_ptr = std::shared_ptr<Node>;

  template <typename... Args>
  Node(Args&&... args) : Gaussian(std::forward<Args>(args)...) {}

  virtual ~Node() = default;

  void send() {
    for (auto neighbor : neighbors_) {
      send(neighbor);
    }
  }

  void send(const shared_ptr& receiver) {
    Gaussian message(*this->prior());

    for (auto it = neighbors_.begin(); it != neighbors_.end(); it++) {
      if (*it != receiver) {
        if (auto potential = this->potential(*it)) {
          message.merge(*potential);
        }
        if (messages_.find(*it) != messages_.end()) {
          message.merge(messages_.at(*it));
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
    messages_.at(sender) = message;
    assert(messages_.at(sender).key() == this->key_);
  }

  auto const& messages() const { return messages_; }

  virtual std::optional<Gaussian> prior() const { return std::nullopt; }

  virtual std::optional<Gaussian> potential(
      const shared_ptr& node = nullptr) = 0;

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

  void update() {
    // update belief
    for (auto const& [_, message] : messages_) {
      assert(message.key() == this->key_);
      this->update({message}, {.type = GaussianMergeType::Step}, nullptr);
    }
  }

  virtual void update(std::vector<Gaussian> messages,
                      UpdateParams params,
                      UpdateResult* result) = 0;

 protected:
  std::map<shared_ptr, Gaussian> messages_;
  std::vector<shared_ptr> neighbors_;
};

}  // namespace gbpc
