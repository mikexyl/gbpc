#ifndef GBPC_VARIABLE_NODE_H_
#define GBPC_VARIABLE_NODE_H_

#include <Eigen/Eigen>
#include <memory>
#include <optional>

#include "gaussian.h"

namespace gbpc {

enum class GaussianMergeType { Merge, Mixture };

template <int Dim> class Variable {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<Variable<Dim>>;

  Variable(Gaussian<Dim> initial) : belief_(initial) {}

  static Gaussian<Dim>
  mixtureGaussian(const Gaussian<Dim> &gauss1, const Gaussian<Dim> &gauss2,
                  std::optional<double> force_alpha = 0.5) {
    double alpha;
    if (force_alpha.has_value()) {
      alpha = force_alpha.value();
    } else {
      if (gauss1.N() == 0 or gauss2.N() == 0) {
        throw std::runtime_error("N is zero");
      }
      alpha = gauss1.N() / (gauss1.N() + gauss2.N());
    }

    typename Gaussian<Dim>::Mu mu_mix =
        alpha * gauss1.mu_ + (1 - alpha) * gauss2.mu_;
    typename Gaussian<Dim>::Sigma mu1mu1t = gauss1.mu_ * gauss1.mu_.transpose();
    typename Gaussian<Dim>::Sigma mu2mu2t = gauss2.mu_ * gauss2.mu_.transpose();
    typename Gaussian<Dim>::Sigma mu_mixmu_mixt = mu_mix * mu_mix.transpose();
    typename Gaussian<Dim>::Sigma Sigma_mix =
        alpha * (gauss1.Sigma_ + mu1mu1t) +
        (1 - alpha) * (gauss2.Sigma_ + mu2mu2t) - mu_mixmu_mixt;
    return Gaussian<Dim>::fromMuSigma(mu_mix, Sigma_mix,
                                      gauss1.N() + gauss2.N());
  }

  void update(const std::vector<Gaussian<Dim>> &messages,
              GaussianMergeType merge_type) {
    switch (merge_type) {
    case GaussianMergeType::Merge: {
      auto eta = belief_.eta_;
      auto lambda = belief_.lambda_;

      for (const auto &message : messages) {
        lambda += message.lambda_;
        eta += message.eta_;
      }

      belief_.eta_ = eta;
      belief_.lambda_ = lambda;
      belief_.updateMoments();
    } break;
    case GaussianMergeType::Mixture: {

      for (const auto &message : messages) {
        belief_ = mixtureGaussian(belief_, message);
      }

    } break;
    default:
      throw std::runtime_error("Unknown GaussianMergeType");
      break;
    }

    // for (const auto &message : messages) {
    //   mu_ = message.mu_;
    //   Sigma_ = message.Sigma_;
    // }
  }

  auto mu() const { return belief_.mu(); }
  auto sigma() const { return belief_.sigma(); }

  auto const &belief() const { return belief_; }
  void relax(double k) { belief_.relax(k); }

protected:
  Gaussian<Dim> belief_;
};

} // namespace gbpc

#endif // GBPC_VARIABLE_NODE_H_
