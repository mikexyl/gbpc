#define CATCH_CONFIG_MAIN
#include <gtsam/geometry/Pose3.h>

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

#include "gbpc/contraction/hellinger.h"

using namespace gtsam;
using namespace gbpc;

// Helper to interpolate between two Pose3s by fraction t in [0,1]
Pose3 interpolatePose(const Pose3& start, const Pose3& end, double t) {
  Vector6 delta = Pose3::Logmap(start.inverse() * end);
  Vector6 interp_delta = t * delta;
  return start * Pose3::Expmap(interp_delta);
}

// Helper to linearly interpolate covariance matrices
Eigen::Matrix<double, 6, 6> interpolateCov(
    const Eigen::Matrix<double, 6, 6>& start,
    const Eigen::Matrix<double, 6, 6>& end,
    double t) {
  return (1.0 - t) * start + t * end;
}

TEST_CASE("Gaussian moving from initial to target distribution over N steps",
          "[hellinger][interpolation]") {
  const size_t N = 10;  // number of interpolation steps

  // Initial pose and covariance
  Pose3 mu0 = Pose3::Identity();
  Eigen::Matrix<double, 6, 6> cov0 = Eigen::Matrix<double, 6, 6>::Identity();

  // Target pose (some translation + rotation)
  Pose3 muT = Pose3(Rot3::RzRyRx(0.3, -0.2, 0.1), Point3(1.0, 0.5, 0.2));
  Eigen::Matrix<double, 6, 6> covT =
      2.0 * Eigen::Matrix<double, 6, 6>::Identity();

  // Generate trajectory vectors
  std::vector<Pose3> means;
  std::vector<Eigen::Matrix<double, 6, 6>> covs;
  for (size_t i = 0; i < N; ++i) {
    double t = static_cast<double>(i) / (N - 1);
    means.push_back(interpolatePose(mu0, muT, t));
    covs.push_back(interpolateCov(cov0, covT, t));
  }

  // Compute metrics along the trajectory
  std::vector<double> hell, alpha, beta;
  Hellinger::computeMetrics(means, covs, hell, alpha, beta);

  // Checks:
  REQUIRE(hell.size() == N - 1);
  REQUIRE(alpha.size() == N - 1);
  REQUIRE(beta.size() == N - 1);

  // Hellinger distances should be >= 0 and <= 1
  for (double h : hell) {
    REQUIRE(h >= 0.0);
    REQUIRE(h <= 1.0);
  }

}