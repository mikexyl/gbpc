#ifndef GBPC_ROBUST_KERNELS_H_
#define GBPC_ROBUST_KERNELS_H_

#include <Eigen/Eigen>
#include <cmath>
#include <iostream>

#include "gbpc/gaussian.h"

namespace gbpc {

template <int Dim> struct Huber {
  using Vector = Eigen::Vector<double, Dim>;
  Huber(Vector gauss_noise_var, Vector mahalanobis_threshold)
      : gauss_noise_var_(gauss_noise_var),
        mahalanobis_threshold_(mahalanobis_threshold),
        adaptive_gauss_noise_var_(gauss_noise_var) {}

  void filter(Gaussian<Dim> *message, Vector dx) {
    auto old_adaptive_gauss_noise_var = adaptive_gauss_noise_var_;
    auto mahalanobis_dist =
        dx.array().abs() / gauss_noise_var_.cwiseSqrt().array();
    Eigen::Array<bool, Eigen::Dynamic, 1> comparison =
        (mahalanobis_dist.array() > mahalanobis_threshold_.array());
    if (comparison.any()) {
      // Compute component-wise operations
      Eigen::VectorXd mahalanobis_dist_sq = mahalanobis_dist.array().square();
      Eigen::VectorXd product_mt_md =
          mahalanobis_threshold_.array() * mahalanobis_dist.array();
      Eigen::VectorXd half_mt_sq =
          0.5 * mahalanobis_threshold_.array().square();
      Eigen::VectorXd denominator = 2 * (product_mt_md - half_mt_sq);

      Eigen::VectorXd m1 = mahalanobis_dist_sq.array() / denominator.array();

      adaptive_gauss_noise_var_ =
          gauss_noise_var_.array() *
          (mahalanobis_dist_sq.array() / denominator.array());
    }

    Eigen::VectorXd weight = old_adaptive_gauss_noise_var.array() /
                             adaptive_gauss_noise_var_.array();

    Eigen::VectorXd lambda_diag = message->lambda_.diagonal();
    Eigen::VectorXd lambda_diag_weighted = lambda_diag.array() * weight.array();

    Vector new_eta = comparison.select(message->eta_.array() * weight.array(),
                                       message->eta_.array());
    Eigen::VectorXd new_lambda_diag =
        comparison.select(lambda_diag_weighted, lambda_diag);

    message->eta_ = new_eta;
    message->lambda_ = new_lambda_diag.asDiagonal();
  }

  Vector gauss_noise_var_;
  Vector mahalanobis_threshold_;
  Vector adaptive_gauss_noise_var_;
};

} // namespace gbpc

#endif // GBPC_ROBUST_KERNELS_H_
