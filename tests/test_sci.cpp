// test_sci_se3_stress.cpp
// Stress-test and wrapper for Split Covariance Intersection on SE(3)

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

using namespace gtsam;
using symbol_shorthand::X;

// -----------------------------------------------------------------------------
// Euclidean SCI on R^6
std::pair<Eigen::Matrix<double,6,1>, Eigen::Matrix<double,6,6>>
SplitCovarianceIntersection(
    const std::vector<Eigen::Matrix<double,6,1>>& delta_v,
    const std::vector<Eigen::Matrix<double,6,6>>& P_v,
    const std::vector<Eigen::Matrix<double,6,6>>& Q_v,
    const Eigen::VectorXd& omega_v)
{
  const size_t N = delta_v.size();
  Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero();
  std::vector<Eigen::Matrix<double,6,6>> A_v(N);
  for (size_t i = 0; i < N; ++i) {
    Eigen::Matrix<double,6,6> S = P_v[i] + omega_v(i) * Q_v[i];
    A_v[i] = S.inverse();
    H.noalias() += omega_v(i) * A_v[i];
  }
  Eigen::Matrix<double,6,6> B = H.inverse();
  Eigen::Matrix<double,6,1> delta_fused = Eigen::Matrix<double,6,1>::Zero();
  for (size_t i = 0; i < N; ++i) {
    Eigen::Matrix<double,6,6> K = omega_v(i) * A_v[i] * B;
    delta_fused.noalias() += K * delta_v[i];
  }
  return {delta_fused, B};
}

// -----------------------------------------------------------------------------
// Manifold-aware SCI wrapper for SE(3)
std::pair<Pose3, Eigen::Matrix<double,6,6>>
SplitCovarianceIntersectionSE3(
    const std::vector<Pose3>& mus,
    const std::vector<Eigen::Matrix<double,6,6>>& covs,
    const Eigen::VectorXd& omega)
{
  const size_t N = mus.size();
  // reference pose = first mean
  Pose3 X_ref = mus[0];
  // log-map into tangent
  std::vector<Eigen::Matrix<double,6,1>> deltas(N);
  for (size_t i = 0; i < N; ++i)
    deltas[i] = Pose3::Logmap(X_ref.inverse() * mus[i]);
  // split covariances
  std::vector<Eigen::Matrix<double,6,6>> P(N), Q(N);
  for (size_t i = 0; i < N; ++i) {
    P[i] = 0.5 * covs[i];
    Q[i] = covs[i] - P[i];
  }
  // fuse in tangent space
  auto [delta_fused, C_fused] =
    SplitCovarianceIntersection(deltas, P, Q, omega);
  // retract to manifold
  Pose3 X_fused = X_ref * Pose3::Expmap(delta_fused);
  return {X_fused, C_fused};
}

// -----------------------------------------------------------------------------
// Project vector onto simplex {omega >= 0, sum omega = 1}
Eigen::VectorXd projectOntoSimplex(Eigen::VectorXd v) {
  const int n = v.size();
  std::vector<double> u(n);
  for (int i = 0; i < n; ++i) u[i] = v[i];
  std::sort(u.begin(), u.end(), std::greater<double>());
  std::vector<double> css(n);
  css[0] = u[0];
  for (int i = 1; i < n; ++i) css[i] = css[i-1] + u[i];
  int rho = -1;
  for (int i = 0; i < n; ++i) {
    double t = (css[i] - 1.0) / (i + 1);
    if (u[i] - t > 0) rho = i;
  }
  double theta = (css[rho] - 1.0) / (rho + 1);
  for (int i = 0; i < n; ++i)
    v[i] = std::max(v[i] - theta, 0.0);
  return v;
}

// -----------------------------------------------------------------------------
// Optimize weights via projected gradient descent on trace objective
Eigen::VectorXd optimizeWeights(
    const std::vector<Pose3>& mus,
    const std::vector<Eigen::Matrix<double,6,6>>& covs,
    int max_iters = 200,
    double alpha = 0.1)
{
  const int N = mus.size();
  Eigen::VectorXd w = Eigen::VectorXd::Ones(N) / N;
  const double eps = 1e-6;
  for (int iter = 0; iter < max_iters; ++iter) {
    double f0 = SplitCovarianceIntersectionSE3(mus, covs, w).second.trace();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(N);
    for (int i = 0; i < N; ++i) {
      Eigen::VectorXd w_eps = w;
      w_eps[i] += eps;
      w_eps = projectOntoSimplex(w_eps);
      double f1 = SplitCovarianceIntersectionSE3(mus, covs, w_eps).second.trace();
      grad[i] = (f1 - f0) / eps;
    }
    w.noalias() -= alpha * grad;
    w = projectOntoSimplex(w);
    alpha *= 0.99;
  }
  return projectOntoSimplex(w);
}

// -----------------------------------------------------------------------------
// Wrapper: optimize ω and perform SCI fusion in one call
std::pair<Pose3, Eigen::Matrix<double,6,6>>
OptimizeAndSplitCovarianceIntersectionSE3(
    const std::vector<Pose3>& mus,
    const std::vector<Eigen::Matrix<double,6,6>>& covs,
    int max_iters = 200,
    double alpha = 0.1)
{
  // 1) optimize weights
  Eigen::VectorXd omega = optimizeWeights(mus, covs, max_iters, alpha);
  // 2) fuse with optimized ω
  return SplitCovarianceIntersectionSE3(mus, covs, omega);
}

// -----------------------------------------------------------------------------
// Main: generate random beliefs, call wrapper, and print results
int main() {
  const int N = 20;
  std::mt19937 rng(42);
  std::normal_distribution<double> nd(0.0, 1.0);

  // generate random Pose3 means and covariances
  std::vector<Pose3> mus;
  std::vector<Eigen::Matrix<double,6,6>> covs;
  for (int i = 0; i < N; ++i) {
    Eigen::Vector3d r(0.2*nd(rng), 0.2*nd(rng), 0.2*nd(rng));
    Rot3 R = Rot3::RzRyRx(r[0], r[1], r[2]);
    Point3 t(1.0*nd(rng), 1.0*nd(rng), 1.0*nd(rng));
    mus.emplace_back(R, t);
    Eigen::Matrix<double,6,6> A;
    for (int r0 = 0; r0 < 6; ++r0)
      for (int c0 = 0; c0 < 6; ++c0)
        A(r0,c0) = nd(rng);
    covs.push_back(A * A.transpose() + 0.01 * Eigen::Matrix<double,6,6>::Identity());
  }

  // call wrapper
  auto [mu_fused, C_fused] =
    OptimizeAndSplitCovarianceIntersectionSE3(mus, covs);

  std::cout << "Fused Pose3:\n" << mu_fused << "\n\n";
  std::cout << "Fused Covariance trace: " << C_fused.trace() << "\n";
  return 0;
}
