#pragma once

#include <gtsam/geometry/Pose3.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "gbpc/contraction/contraction.h"

using namespace gtsam;

namespace gbpc {
struct KLDivergence : Contraction {
  template <typename... Args>
  KLDivergence(Args&&... args) : Contraction(std::forward<Args>(args)...) {}

  std::optional<Gaussian> operator()(const Gaussian& curr,
                                     const Gaussian& next) override {
    float dxy_no_eta =
        computeKLDivergence(curr.mu(), next.mu(), curr.Sigma(), next.Sigma());
    float d_yx =
        computeKLDivergence(next.mu(), curr.mu(), next.Sigma(), curr.Sigma());
    Gaussian x_diff = next - curr;
    auto Sigma_d = x_diff.Sigma();
    float gamma = params_.gamma;
    float d_reset = params_.d_reset;
    auto d_xy_curr = curr.dxycurr();
    auto dxy = curr.dxy();
    float rate = dxy_no_eta / d_xy_curr;
    float alpha = 0;
    if (gamma < 0) {
      alpha = params_.contract_alpha;
    } else {
      alpha = 1 / (1 + gamma * rate);
    }
    if (dxy_no_eta < 0) {
      dxy_no_eta = 0;
      std::cerr << fmt::format(
          "d_tau_x_tau_y_({}) is negative, {},{} \n", dxy_no_eta, dxy, rate);
    }
    if (d_yx > d_reset) {
      spdlog::warn("alpha({}) is not between 0 and 1, {},{},{}",
                   alpha,
                   dxy_no_eta,
                   d_xy_curr,
                   rate);
      Gaussian reset_gauss(next);
      reset_gauss.dxy() = std::numeric_limits<float>::max();
      reset_gauss.dxycurr() = dxy_no_eta;
      return reset_gauss;
    }

    if (not(alpha > -std::numeric_limits<float>::epsilon() &&
            alpha < 1 + std::numeric_limits<float>::epsilon())) {
      spdlog::warn("alpha({}) is not between 0 and 1, {},{},{}",
                   alpha,
                   dxy_no_eta,
                   d_xy_curr,
                   rate);
      Gaussian reset_gauss = curr;
      reset_gauss.Sigma().setIdentity();
      reset_gauss.Sigma() *= 1e4;  // reset to large covariance
      reset_gauss.dxy() = std::numeric_limits<float>::max();
      reset_gauss.dxycurr() = dxy_no_eta;
      return reset_gauss;
    }

    Gaussian result = curr;
    result.contractionRate() = rate;
    // clip alpha to be between 0.8 and 0.99
    alpha = std::fmax(0.0, std::fmin(1.0, alpha));
    float d_target = dxy * alpha;

    if (dxy_no_eta <= d_target) {
      result = next;  // already converging slow enough
      result.dxy() = dxy_no_eta;
      result.dxycurr() = dxy_no_eta;
      return result;
    }

    auto mu_d = x_diff.mu();
    auto Sigma_1 = curr.Sigma();

    float diff = mu_d.transpose() * Sigma_1.inverse() * mu_d;
    // TODO(mikexyl): make this an option
    float tr = (Sigma_1.inverse() * Sigma_d).trace();
    float denom = diff + tr;

    float lambda;
    if (denom < 1e-6) {
      lambda = 1;
    } else {
      lambda = std::sqrt(2 * d_target / denom);
    }

    lambda = std::fmin(lambda, 1);

    result.mu() = traits<VALUE>::Logmap(traits<VALUE>::Retract(
        traits<VALUE>::Expmap(curr.mu()), mu_d * lambda));
    result.Sigma() =
        result.Sigma() + TransformCovariance<VALUE>(
                             traits<VALUE>::Expmap(mu_d * lambda))(Sigma_d) *
                             lambda * lambda;
    result.contractionStepSize() = lambda;
    result.dxy() = d_target;
    result.dxycurr() = dxy_no_eta;
    return result;
  }

  static double computeKLDivergence(const Eigen::VectorXd& mu1,
                                    const Eigen::VectorXd& mu2,
                                    const Eigen::MatrixXd& cov1,
                                    const Eigen::MatrixXd& cov2) {
    // Check that dimensions match
    if (mu1.size() != mu2.size()) {
      throw std::invalid_argument(
          fmt::format("Means must be of the same dimension, but got {} and {}",
                      mu1.size(),
                      mu2.size()));
    }
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

  // Compute the Kullback-Leibler divergence between two 6D Gaussian
  // distributions (Pose3 + 6x6 cov)
  static double klDivergenceGaussian(
      const Vector6& delta,  // logmap(mu1.inverse() * mu2)
      const Eigen::Matrix<double, 6, 6>& Sigma1,
      const Eigen::Matrix<double, 6, 6>& Sigma2) {
    Eigen::Matrix<double, 6, 6> Sigma_avg = 0.5 * (Sigma1 + Sigma2);
    double det1 = Sigma1.determinant();
    double det2 = Sigma2.determinant();
    double det_avg = Sigma_avg.determinant();

    if (det1 <= 0 || det2 <= 0 || det_avg <= 0) {
      throw std::runtime_error("Covariance matrix not positive definite.");
    }

    double exp_term =
        exp(-0.125 * delta.transpose() * Sigma_avg.inverse() * delta);
    double BC = pow(det1, 0.25) * pow(det2, 0.25) / sqrt(det_avg) * exp_term;
    double KL = -log(BC);
    return KL;
  }
};

}  // namespace gbpc