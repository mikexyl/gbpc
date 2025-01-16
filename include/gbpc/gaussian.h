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
#include <spdlog/spdlog.h>

#include <Eigen/Eigen>
#include <concepts>
#include <optional>

namespace gtsam {
class NoiseModelValue {};
}  // namespace gtsam

using namespace gtsam;

namespace gbpc {

enum class GaussianMergeType {
  Merge = 0,
  MergeRobust,
  Average,
  Replace,
  Step,
  Contract,
  ContractPertSigma,
  ContractBoundedSigma
};

struct UpdateParams {
  GaussianMergeType type = GaussianMergeType::Average;
  double relax = 1.0;
  bool use_fixed_alpha = false;
  double fixed_alpha = 0;
  float belief_change_threshold = 0.0;
  double step_size = 0.001;
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

  Gaussian() : N_(0) {}
  explicit Gaussian(Key key) : key_(key) {}
  Gaussian(const Gaussian& other) = default;
  Gaussian(Gaussian&& other) = default;
  Gaussian(Key key,
           const Vector& mu,
           const Vector& eta,
           const Eigen::MatrixXd& Sigma,  // covariance
           const Eigen::MatrixXd& lambda,
           size_t N)
      : mu_(mu), eta_(eta), Sigma_(Sigma), lambda_(lambda), N_(N), key_(key) {}
  Gaussian(Key key, const Vector& mu, const Eigen::MatrixXd& Sigma, size_t N)
      : mu_(mu), Sigma_(Sigma), N_(N), key_(key) {
    updateCanonical();
  }

  static Gaussian Random(Key key, size_t dim, size_t N = 1) {
    Vector mu = Vector::Random(dim);
    Matrix Sigma = Matrix::Random(dim, dim) * 100;
    return Gaussian(key, mu, Sigma, N);
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

  Gaussian operator-(const This& other) const {
    return Gaussian(key_, mu_ - other.mu_, Sigma_ + other.Sigma_, N_);
  }

  double hellingerDistance(const This& other) const {
    return hellingerDistance(mu_, other.mu_, Sigma_, other.Sigma_);
  }

  double KLDivergence(const This& other) const {
    return KLDivergence(mu_, other.mu_, Sigma_, other.Sigma_);
  }

  static double KLDivergence(const Eigen::VectorXd& mu1,
                             const Eigen::VectorXd& mu2,
                             const Eigen::MatrixXd& cov1,
                             const Eigen::MatrixXd& cov2) {
    // Check that dimensions match
    assert(mu1.size() == mu2.size() && "Means must be of the same dimension");
    assert(cov1.rows() == cov1.cols() && cov2.rows() == cov2.cols() &&
           cov1.rows() == mu1.size() &&
           "Covariances must be square and match mean dimensions");

    int k = mu1.size();

    // Compute the inverse and determinant of cov2
    Eigen::MatrixXd cov2_inv = cov2.inverse();
    double det_cov1 = cov1.determinant();
    double det_cov2 = cov2.determinant();

    // Compute the trace term
    double trace_term = (cov2_inv * cov1).trace();

    // Compute the quadratic term
    Eigen::VectorXd diff = mu2 - mu1;
    double quadratic_term = diff.transpose() * cov2_inv * diff;

    // Compute the log-determinant term
    double log_det_term = std::log(det_cov2 / det_cov1);

    // Calculate the KL divergence
    double kl_divergence =
        0.5 * (trace_term + quadratic_term - k + log_det_term);

    return kl_divergence;
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

  void shiftMu(const Vector& shift) {
    mu_ += shift;
    updateCanonical();
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
        return gauss2;
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
    size_t weighted_N =
        (N1 * N1 + N2 * N2) / (N1 + N2 + 1);  // + 1 to avoid division by zero

    return Gaussian(key, mu_mix, Sigma_mix, weighted_N);
  }

  void merge(const Gaussian& other, bool expect_same_key = true) {
    if (N_ == 0) {
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

  bool empty() const { return mu_.size() == 0; }

  void updateMu(const Vector& mu) {
    mu_ = mu;
    updateCanonical();
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

  void send() {
    for (auto neighbor : neighbors_) {
      send(neighbor);
    }
  }

  void send(const shared_ptr& receiver) {
    Gaussian message(this->prior());

    for (auto it = neighbors_.begin(); it != neighbors_.end(); it++) {
      if (*it != receiver) {
        if (auto potential = this->potential(*it)) {
          message.merge(*potential);
        }
        if (messages_.find(*it) != messages_.end()) {
          message.merge(messages_[*it]);
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

  Belief(const Key& key)
      : Node(key),
        d_xy_(std::numeric_limits<float>::max()),
        d_sigma_(std::numeric_limits<float>::max()) {}

  Belief(const This& other) = default;
  Belief(const Gaussian& other)
      : Node(other), d_xy_(std::numeric_limits<float>::max()) {}

  Belief(Key key, const Vector& mu, const Covariance& Sigma, size_t N)
      : Node(Gaussian(key, mu, Sigma, N)),
        d_xy_(std::numeric_limits<float>::max()) {}

  virtual std::optional<Gaussian> potential(
      const Node::shared_ptr& node = nullptr) override {
    return std::nullopt;
  }

  VALUE value() const { return traits<VALUE>::Expmap(mu_); }

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

  void step(This const& other, double step_size) {
    auto mu_node = traits<VALUE>::Expmap(mu_);
    auto mu_fn = traits<VALUE>::Expmap(other.mu_);

    typename VALUE::Jacobian H_tau_mu_fn, H_tau_mu_node;
    auto tau0_fn =
        traits<VALUE>::Between(mu_node, mu_fn, H_tau_mu_node, H_tau_mu_fn);

    if (tau0_fn.equals(traits<VALUE>::Identity(), 1e-3)) {
      return;
    }

    Eigen::Matrix<double, 6, 6> Lambda0_fn =
        H_tau_mu_fn.transpose() * other.Lambda() * H_tau_mu_fn;
    Eigen::Matrix<double, 6, 1> tau0_fn_ = traits<VALUE>::Logmap(tau0_fn);
    typename VALUE::TangentVector tau_plus = Lambda0_fn * tau0_fn_ * step_size;
    auto mu_node_new = traits<VALUE>::Retract(mu_node, tau_plus);
    auto Lambda_plus = Lambda0_fn;

    typename VALUE::Jacobian H_mu_new_tau_plus, H_mu_new_mu_0;
    mu_node_new = traits<VALUE>::Retract(
        mu_node, tau_plus, H_mu_new_mu_0, H_mu_new_tau_plus);

    auto Lambda_new =
        H_mu_new_tau_plus.transpose() * Lambda_plus * H_mu_new_tau_plus;

    mu_ = traits<VALUE>::Logmap(mu_node_new);
    Sigma_ = Lambda_new.inverse();

    this->updateCanonical();
  }

  Gaussian inverse(std::optional<Key> key) const {
    auto new_key = this->key();
    if (key.has_value()) {
      new_key = key.value();
    }

    auto value = traits<VALUE>::Expmap(mu_);
    typename VALUE::Jacobian J;
    auto inverse_mu = traits<VALUE>::Logmap(value.inverse(J));

    auto new_Sigma = J * Sigma_ * J.transpose();

    return Gaussian(new_key, inverse_mu, new_Sigma, N_);
  }

  void update(std::vector<Gaussian> messages,
              UpdateParams params,
              UpdateResult* result) override {
    // relax all message
    for (auto& message : messages) {
      message.relax(params.relax);
    }

    switch (params.type) {
      case GaussianMergeType::Merge:
      case GaussianMergeType::MergeRobust: {
        for (auto message : messages) {
          float distance = this->hellingerDistance(message);
          if (distance < params.belief_change_threshold) {
            result->status.push_back(UpdateResult::Failed);
            continue;
          } else {
            result->status.push_back(UpdateResult::Success);
          }

          if (params.type == GaussianMergeType::MergeRobust) {
            double hellinger = this->hellingerDistance(message);
            double k = std::max(0.1, 1 - hellinger);
            message.relax(k);
          }
          this->merge(message);
        }
      } break;
      case GaussianMergeType::Average: {
        for (const auto& message : messages) {
          // TODO: ! this is not on manifold
          auto message_copy = message;
          message_copy.relax(params.relax);

          float diff = message.KLDivergence(*this);
          result->change.push_back(diff);
          if (diff < params.belief_change_threshold) {
            result->status.push_back(UpdateResult::Failed);
            continue;
          } else {
            result->status.push_back(UpdateResult::Success);
          }

          this->replace(mixtureGaussian(
              *this,
              message_copy,
              params.use_fixed_alpha ? std::optional<double>(params.fixed_alpha)
                                     : std::nullopt));
        }
      } break;
      case GaussianMergeType::Replace: {
        auto message = messages.front();
        this->replace(message);
        result->change.push_back(message.KLDivergence(*this));
        result->status.push_back(UpdateResult::Success);
      } break;
      case GaussianMergeType::Step: {
        auto message = messages.front();
        this->step(message, params.step_size);
        result->change.push_back(message.KLDivergence(*this));
        result->status.push_back(UpdateResult::Success);
      } break;
      case GaussianMergeType::Contract: {
        auto message = messages.front();
        this->contract(message, false, false);
        result->change.push_back(message.KLDivergence(*this));
        result->status.push_back(UpdateResult::Success);
      } break;
      case GaussianMergeType::ContractPertSigma: {
        auto message = messages.front();
        this->contract(message, true, false);
        result->change.push_back(message.KLDivergence(*this));
        result->status.push_back(UpdateResult::Success);
      } break;
      case GaussianMergeType::ContractBoundedSigma: {
        auto message = messages.front();
        this->contract(message, true, true);
        result->change.push_back(message.KLDivergence(*this));
        result->status.push_back(UpdateResult::Success);
      } break;
      default:
        throw std::runtime_error("Unknown GaussianMergeType");
        break;
    }
  }

  static double Chi2(Gaussian z1, Gaussian z2) {
    auto d_mu = z1.mu() - z2.mu();
    auto sigma1 = z1.Sigma();
    auto sigma2 = z2.Sigma();
    auto I = sigma1;
    I.setIdentity();
    double chi = d_mu.transpose() * sigma2.inverse() * d_mu +
                 std::pow((sigma2.inverse() * sigma1 - I).trace(), 2);
    return chi;
  }

  void contract(const Gaussian& other,
                bool use_pert_sigma = true,
                bool bound_sigma = false) {
    float d_tau_x_tau_y_ = this->KLDivergence(other);
    Gaussian x_diff = other - (*this);

    float rate = d_tau_x_tau_y_ / d_xy_;
    float gamma = 0.1;
    float alpha = 1 / (1 + gamma * rate);

    double chi2 = Chi2(*this, other) + Chi2(other, *this) + 1e-6;
    chi2 /= 2.0;

    // grad_new must be smaller than grad_old_
    float d_target = d_xy_ * alpha;
    float d_t_sigma = d_sigma_ * alpha;
    if (d_tau_x_tau_y_ < 1e-6) {
      return;
    }
    if (d_tau_x_tau_y_ <= d_target or d_tau_x_tau_y_ > 3.0) {
      this->replace(other);
      d_xy_ = d_tau_x_tau_y_;
      d_sigma_ = other.Sigma().norm() - Sigma_.norm();
      spdlog::debug("initial d_xy_: {}", d_xy_);
      return;
    }

    auto mu_d = x_diff.mu();
    auto Sigma_d = x_diff.Sigma();
    auto Sigma_1 = this->Sigma();

    float diff = mu_d.transpose() * Sigma_1.inverse() * mu_d;
    float tr = 0;
    if (use_pert_sigma) {
      tr = (Sigma_1.inverse() * Sigma_d).trace();
    }
    float denom = diff + tr;

    float lambda;
    if (denom < 1e-6) {
      lambda = 1;
    } else {
      lambda = std::sqrt(2 * d_target / denom);
    }

    lambda = std::fmin(lambda, 1);

    this->mu_ = traits<VALUE>::Logmap(traits<VALUE>::Retract(
        traits<VALUE>::Expmap(this->mu()), mu_d * lambda));

    double k_sigma = 1.;
    if (bound_sigma) {
      double d_sigma = (Sigma_d).norm() * lambda * lambda;
      if (d_sigma > d_t_sigma) {
        k_sigma = d_t_sigma / d_sigma;
      }
    }

    this->Sigma_ =
        this->Sigma() + TransformCovariance<VALUE>(
                            traits<VALUE>::Expmap(mu_d * lambda))(Sigma_d) *
                            lambda * lambda * k_sigma;

    spdlog::debug("lambda: {}", lambda);

    updateCanonical();

    d_xy_ = d_target;
    d_sigma_ = d_t_sigma;
  }

 protected:
  float d_xy_;
  float d_sigma_;
  static constexpr float kAlpha = 0.8;
};

}  // namespace gbpc

#endif  // GBPC_GAUSSIAN_H_
