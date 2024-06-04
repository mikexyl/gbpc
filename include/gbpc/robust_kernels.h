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
    std::cout << "------------" << std::endl;
    std::cout << "dx: " << dx.transpose() << std::endl;
    auto old_adaptive_gauss_noise_var = adaptive_gauss_noise_var_;
    auto mahalanobis_dist =
        dx.array().abs() / gauss_noise_var_.cwiseSqrt().array();
    std::cout << mahalanobis_dist.transpose() << std::endl;
    Eigen::Array<bool, Eigen::Dynamic, 1> comparison =
        (mahalanobis_dist.array() > mahalanobis_threshold_.array());
    std::cout << comparison.transpose() << std::endl;
    if (comparison.any()) {
      // Compute component-wise operations
      Eigen::VectorXd mahalanobis_dist_sq = mahalanobis_dist.array().square();
      std::cout << mahalanobis_dist_sq.transpose() << std::endl;
      Eigen::VectorXd product_mt_md =
          mahalanobis_threshold_.array() * mahalanobis_dist.array();
      std::cout << product_mt_md.transpose() << std::endl;
      Eigen::VectorXd half_mt_sq =
          0.5 * mahalanobis_threshold_.array().square();
      std::cout << half_mt_sq.transpose() << std::endl;
      Eigen::VectorXd denominator = 2 * (product_mt_md - half_mt_sq);
      std::cout << denominator.transpose() << std::endl;

      Eigen::VectorXd m1 = mahalanobis_dist_sq.array() / denominator.array();
      std::cout << m1.transpose() << std::endl;

      adaptive_gauss_noise_var_ =
          gauss_noise_var_.array() *
          (mahalanobis_dist_sq.array() / denominator.array());
      std::cout << adaptive_gauss_noise_var_.transpose() << std::endl;
    }

    Eigen::VectorXd weight = old_adaptive_gauss_noise_var.array() /
                             adaptive_gauss_noise_var_.array();
    std::cout << weight.transpose() << std::endl;

    Eigen::VectorXd lambda_diag = message->lambda_.diagonal();
    Eigen::VectorXd lambda_diag_weighted = lambda_diag.array() * weight.array();

    Vector new_eta = comparison.select(message->eta_.array() * weight.array(), message->eta_.array());
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