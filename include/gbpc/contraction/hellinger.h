#pragma once

#include <gtsam/geometry/Pose3.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "gbpc/contraction/contraction.h"

using namespace std;
using namespace gtsam;

static constexpr double kHellingerEpsilon = 1e-12;

namespace gbpc {

struct Hellinger : Contraction {
  template <typename... Args>
  Hellinger(Args&&... args) : Contraction(std::forward<Args>(args)...) {}

  Gaussian operator()(const Gaussian& curr, const Gaussian& next) override {
    Gaussian result(curr);

    Pose3 p_curr = Pose3::Expmap(curr.mu()), p_next = Pose3::Expmap(next.mu());

    Matrix66 Ad_inv = p_curr.inverse().AdjointMap();

    Matrix66 cov_curr_in_pcurr = Ad_inv * curr.Sigma() * Ad_inv.transpose();
    Matrix66 cov_next_in_pcurr = Ad_inv * next.Sigma() * Ad_inv.transpose();

    double contraction_alpha = params_.contract_alpha;

    std::vector<double> hell, alpha, beta;
    std::vector<Pose3> means{p_curr, p_next};  // means in Pose3 space
    std::vector<Eigen::Matrix<double, 6, 6>> covs{
        cov_curr_in_pcurr, cov_next_in_pcurr};  // covariances in Pose3 space
    computeMetrics(means, covs, hell, alpha, beta);

    // check if hell.(0) is NaN or Inf
    if (std::isnan(hell.at(0)) ||
        std::isinf(hell.at(0))) {  // hellinger distance is NaN or Inf
      throw std::runtime_error("Hellinger distance is NaN or Inf: " +
                               std::to_string(hell.at(0)));
    }
    // check if result.dxycurr() is NaN or Inf
    if (std::isnan(curr.dxycurr()) || std::isinf(curr.dxycurr())) {
      throw std::runtime_error("Current dxycurr is NaN or Inf: " +
                               std::to_string(curr.dxycurr()));
    }

    // if contraction_alpha is negative, we compute it based on the convergence
    // rate
    result.dxycurr() = std::fmin(1.0, curr.dxycurr());
    result.dxycurr() = std::fmax(kHellingerEpsilon, result.dxycurr());
    double rate = 0;
    if (contraction_alpha < 0) {
      rate = hell.at(0) / result.dxycurr();
      contraction_alpha = 1 / (1 + params_.gamma * rate);
      // check if rate is NaN or Inf
      if (std::isnan(rate) || std::isinf(rate)) {
        throw std::runtime_error("Convergence rate is NaN or Inf: " +
                                 std::to_string(rate));
      }
    }

    result.dxycurr() = hell.at(0);

    if (hell.at(0) > params_.d_reset) {
      Gaussian reset_gauss(next);
      reset_gauss.dxy() = 1.0;
      reset_gauss.dxycurr() = hell.at(0);
      return reset_gauss;
    }

    // clamp hellinger distance to [0, 1], so we can handle default dxy to inf,
    // which is the setting of KLD
    result.dxy() = std::fmin(1.0, result.dxy());

    // check hellinger in [0, 1]
    if (hell.at(0) < 0 || hell.at(0) > 1) {
      throw std::runtime_error("Hellinger distance out of bounds [0, 1]: " +
                               std::to_string(hell.at(0)));
    }

    // check contract alpha in [0, 1]
    if (contraction_alpha < 0 || contraction_alpha > 1) {
      throw std::runtime_error("Contract alpha out of bounds [0, 1]: " +
                               std::to_string(contraction_alpha));
    }

    double target_hellinger = result.dxy() * contraction_alpha;
    if (target_hellinger < 0 || target_hellinger > 1) {
      throw std::runtime_error(
          fmt::format("Target Hellinger distance out of bounds [0, 1]: {}, "
                      "dxy: {}, contract alpha: {}",
                      target_hellinger,
                      result.dxy(),
                      contraction_alpha));
    }

    if (result.dxycurr() < target_hellinger) {
      // already converging slow enough
      result = next;
      result.dxy() = hell.at(0);
      result.dxycurr() = hell.at(0);
      return result;
    }

    double D_star = std::log(1 - target_hellinger * target_hellinger);

    // check alpha < 0
    if (alpha.at(0) > 0) {
      throw std::runtime_error(
          "Alpha metric is positive, which is unexpected: " +
          std::to_string(alpha.at(0)));
    }

    if (D_star > 0 or std::isnan(D_star) or std::isinf(D_star)) {
      throw std::runtime_error(
          fmt::format("D_star is non-negative, which is unexpected: {}, target "
                      "Hellinger {}, contract alpha: {}, rate: {}",
                      D_star,
                      target_hellinger,
                      contraction_alpha,
                      rate));
    }

    double epsilon = 0.0;
    if (D_star != 0) {
      epsilon = computeStepSize(alpha.at(0), beta.at(0), D_star);
    }
    if (std::isnan(epsilon) || std::isinf(epsilon) || epsilon < 0) {
      throw std::runtime_error(
          fmt::format("Invalid step size epsilon: {}, alpha: {}, beta: {}, "
                      "D_star: {}, contract alpha: {}",
                      epsilon,
                      alpha.at(0),
                      beta.at(0),
                      D_star,
                      contraction_alpha));
    }

    // clamp epsilon to [0, 1]
    epsilon = std::max(0.0, std::min(1.0, epsilon));

    auto mu_d = Pose3::Logmap(p_curr.inverse() * p_next);
    // TODO(mikexyl): should transform Sigma first
    Matrix6 Delta_Sigma = cov_next_in_pcurr - cov_curr_in_pcurr;

    result.mu() = traits<VALUE>::Logmap(traits<VALUE>::Retract(
        traits<VALUE>::Expmap(curr.mu()), mu_d * epsilon));
    Matrix6 Delta_Sigma_transformed = p_curr.AdjointMap() * Delta_Sigma *
                                      p_curr.AdjointMap().transpose() *
                                      epsilon * epsilon;
    result.Sigma() += Delta_Sigma_transformed;
    result.contractionStepSize() = epsilon;
    result.dxy() = target_hellinger;
    result.dxycurr() = hell.at(0);
    return result;
  }

  // Compute the Hellinger distance between two 6D Gaussian distributions (Pose3
  // + 6x6 cov)
  static double hellingerDistanceGaussian(
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
    double H =
        sqrt(std::max(0.0, 1.0 - BC));  // clip negative due to numerical errors
    return H;
  }

  // Compute alpha and beta metrics between two Gaussians (in tangent space)
  static void computeAlphaBeta(const Vector6& delta,
                               const Eigen::Matrix<double, 6, 6>& Sigma1,
                               const Eigen::Matrix<double, 6, 6>& Sigma2,
                               double& alpha,
                               double& beta) {
    Eigen::Matrix<double, 6, 6> C_inv = Sigma1.inverse();
    alpha = -(1. / 8.) * delta.transpose() * C_inv * delta;

    Eigen::Matrix<double, 6, 6> delta_C = Sigma2 - Sigma1;
    Eigen::Matrix<double, 6, 6> M = C_inv * delta_C;

    double trace_term = (M * M).trace();
    double quadratic_term = delta.transpose() * C_inv * delta_C * C_inv * delta;
    // double quadratic_term = 0;

    beta = -(1. / 16.) * (trace_term - quadratic_term);  // -1/16 = -0.0625
  }

  // Main function computing Hellinger, alpha, beta for a sequence of Pose3
  // means and covariances
  static void computeMetrics(const vector<Pose3>& means,
                             const vector<Eigen::Matrix<double, 6, 6>>& covs,
                             vector<double>& hell,
                             vector<double>& alpha,
                             vector<double>& beta) {
    size_t N = means.size();
    if (covs.size() != N) {
      throw std::runtime_error("Means and covariances vector size mismatch.");
    }

    hell.resize(N - 1);
    alpha.resize(N - 1);
    beta.resize(N - 1);

    for (size_t k = 0; k < N - 1; ++k) {
      // Compute the 6D tangent difference between poses (log map)
      Vector6 delta = Pose3::Logmap(means[k].inverse() * means[k + 1]);

      hell[k] = hellingerDistanceGaussian(delta, covs[k], covs[k + 1]);
      computeAlphaBeta(delta, covs[k], covs[k + 1], alpha[k], beta[k]);
    }
  }

  static double computeStepSize(double alpha, double beta, double D_star) {
    // Threshold for considering beta zero (to avoid numerical issues)
    const double tol = kHellingerEpsilon;

    int beta_sign = (beta > 0) - (beta < 0);  // -1 if beta < 0, 1 if beta > 0

    // Handle beta != 0 case
    if (std::abs(beta) > tol) {
      double discriminant = alpha * alpha + 4.0 * beta * D_star;
      if (discriminant < 0) {
        std::cerr << "Warning: Discriminant is negative, returning NaN"
                  << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
      }
      double numerator = -alpha + beta_sign * std::sqrt(discriminant);
      double denominator = 2.0 * beta;

      // Check for division by zero or negative inside sqrt
      if (denominator == 0.0 || numerator / denominator < 0.0) {
        std::cerr
            << "Warning: Invalid value under sqrt for epsilon, returning NaN"
            << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
      }

      return std::sqrt(numerator / denominator);
    } else {
      // Handle beta == 0 (or nearly zero) case
      if (alpha == 0.0 || D_star / alpha < 0.0) {
        return 0;
      }
      return std::sqrt(D_star / alpha);
    }
  }
};
}  // namespace gbpc