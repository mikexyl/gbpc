#pragma once

#include "gbpc/contraction/hellinger.h"
#include "gbpc/contraction/kl_divergence.h"
#include "gbpc/gaussian.h"

namespace gbpc {

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

  template <typename... Args>
  Belief(Args&&... args) : Node(std::forward<Args>(args)...) {}

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

    return Gaussian(new_key, inverse_mu, new_Sigma, degree_);
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
          if (params.type == GaussianMergeType::MergeRobust) {
            throw std::runtime_error("MergeRobust deleted");
          }
          this->merge(message);
        }
      } break;
      case GaussianMergeType::Damp: {
        for (const auto& message : messages) {
          // TODO: ! this is not on manifold
          auto message_copy = message;
          message_copy.relax(params.relax);
          this->replace(Damp(*this,
                             message_copy,
                             params.use_fixed_alpha
                                 ? std::optional<double>(params.fixed_alpha)
                                 : std::nullopt));
        }
      } break;
      case GaussianMergeType::Replace: {
        auto message = messages.front();
        this->replace(message);
        result->status.push_back(UpdateResult::Success);
      } break;
      case GaussianMergeType::Step: {
        auto message = messages.front();
        this->step(message, params.step_size);
        result->status.push_back(UpdateResult::Success);
      } break;
      case gbpc::GaussianMergeType::DampContract: {
        auto message = messages.front();
        auto damped_message = Gaussian::Damp(*this, message, 0.5);
        this->contract(damped_message, params, result);
      } break;
      case GaussianMergeType::Contract: {
        auto message = messages.front();
        this->contract(message, params, result);
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
                UpdateParams params = {},
                UpdateResult* result = nullptr) {
    std::optional<Gaussian> result_gaussian;
    switch (params.metric_type) {
      case MetricType::Hellinger: {
        Hellinger hellinger(params);
        result_gaussian = hellinger(*this, other);
      } break;
      case MetricType::KLDivergence: {
        KLDivergence kl_divergence(params);
        result_gaussian = kl_divergence(*this, other);
      } break;
      case MetricType::Chi2: {
        throw std::runtime_error("Chi2 contraction is not implemented.");
      } break;
      default:
        throw std::runtime_error("Unknown MetricType");
    }

    if (!result_gaussian.has_value()) {
      if (result) {
        result->status.push_back(UpdateResult::Failed);
        // std::cout << "Contract failed for key: "
                  // << DefaultKeyFormatter(this->key()) << std::endl;
        result_gaussian = other;
      }
    } else {
      if (result) {
        result->status.push_back(UpdateResult::Success);
      }
    }

    this->mu() = result_gaussian->mu();
    this->Sigma() = result_gaussian->Sigma();
    this->contractionStepSize() = result_gaussian->contractionStepSize();
    this->contractionRate() = result_gaussian->contractionRate();
    this->dxycurr() = result_gaussian->dxycurr();
    this->dxy() = result_gaussian->dxy();
    updateCanonical();
  }
};

}  // namespace gbpc