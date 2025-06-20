#pragma once

#include <gtsam/geometry/Pose3.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "gbpc/contraction/contraction.h"

using namespace std;
using namespace gtsam;

namespace gbpc {

struct Hellinger : Contraction {
  template <typename... Args>
  Hellinger(Args&&... args) : Contraction(std::forward<Args>(args)...) {}

  Gaussian operator()(const Gaussian& curr, const Gaussian& next) override {
    Gaussian result(curr);

    std::cout << "curr sigma: \n" << curr.Sigma() << std::endl;
    std::cout << "next sigma: \n" << next.Sigma() << std::endl;

    std::vector<double> hell, alpha, beta;
    std::vector<Pose3> means{Pose3::Expmap(curr.mu()),
                             Pose3::Expmap(next.mu())};
    std::vector<Eigen::Matrix<double, 6, 6>> covs{curr.Sigma(), next.Sigma()};
    computeMetrics(means, covs, hell, alpha, beta);

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
    if (params_.contract_alpha < 0 || params_.contract_alpha > 1) {
      throw std::runtime_error("Contract alpha out of bounds [0, 1]: " +
                               std::to_string(params_.contract_alpha));
    }

    double target_hellinger = result.dxy() * params_.contract_alpha;
    if (target_hellinger < 0 || target_hellinger > 1) {
      throw std::runtime_error(
          fmt::format("Target Hellinger distance out of bounds [0, 1]: {}, "
                      "dxy: {}, contract alpha: {}",
                      target_hellinger,
                      result.dxy(),
                      params_.contract_alpha));
    }

    if (result.dxycurr() < target_hellinger) {
      // already converging slow enough
      result = next;
      result.dxy() = hell.at(0);
      result.dxycurr() = hell.at(0);
      std::cout << "epsilon 1" << std::endl;
      return result;
    }

    double D_star = std::log(1 - target_hellinger * target_hellinger);

    // check alpha < 0
    if (alpha.at(0) > 0) {
      throw std::runtime_error(
          "Alpha metric is positive, which is unexpected: " +
          std::to_string(alpha.at(0)));
    }
    // check beta < 0
    if (beta.at(0) > 0) {
      throw std::runtime_error(
          "Beta metric is positive, which is unexpected: " +
          std::to_string(beta.at(0)));
    }
    // check D_star < 0
    if (D_star > 0) {
      throw std::runtime_error(
          fmt::format("D_star is non-negative, which is unexpected: {}, target "
                      "Hellinger {}",
                      D_star,
                      target_hellinger));
    }

    double epsilon = 0.0;
    if (D_star != 0) {
      epsilon = computeStepSize(alpha.at(0), beta.at(0), D_star);
    }
    if (std::isnan(epsilon) || std::isinf(epsilon) || epsilon < 0) {
      throw std::runtime_error(
          fmt::format("Invalid step size epsilon: {}, alpha: {}, beta: {}, "
                      "D_star: {}",
                      epsilon,
                      alpha.at(0),
                      beta.at(0),
                      D_star));
    }

    // clamp epsilon to [0, 1]
    epsilon = std::max(0.0, std::min(1.0, epsilon));

    auto pose_curr = Pose3::Expmap(curr.mu());
    auto pose_next = Pose3::Expmap(next.mu());
    auto mu_d = Pose3::Logmap(pose_curr.inverse() * pose_next);
    // TODO(mikexyl): should transform Sigma first
    Matrix6 Delta_Sigma = next.Sigma() - curr.Sigma();

    std::cout << "Delta_Sigma: \n" << Delta_Sigma << std::endl;
    std::cout << mu_d.transpose() << std::endl;
    std::cout << epsilon << std::endl;
    // std::cout << result.Sigma() << std::endl;

    result.mu() = traits<VALUE>::Logmap(traits<VALUE>::Retract(
        traits<VALUE>::Expmap(curr.mu()), mu_d * epsilon));
    Matrix6 Delta_Sigma_transformed = TransformCovariance<VALUE>(
        traits<VALUE>::Expmap(mu_d * epsilon))(Delta_Sigma);
    result.Sigma() =
        result.Sigma() + Delta_Sigma_transformed * epsilon * epsilon;
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
    // double quadratic_term = delta.transpose() * C_inv * delta_C * C_inv *
    // delta;
    double quadratic_term = 0;

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
    const double tol = 1e-12;

    // Handle beta != 0 case
    if (std::abs(beta) > tol) {
      double discriminant = alpha * alpha + 4.0 * beta * D_star;
      if (discriminant < 0) {
        std::cerr << "Warning: Discriminant is negative, returning NaN"
                  << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
      }
      double numerator = -alpha - std::sqrt(discriminant);
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