#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include "gbpc/factor.h"
#include "gbpc/variable.h"

using namespace gbpc;
using namespace gtsam;

class GroupOpsPose3 {
public:
  using Vector = Eigen::Vector<double, 6>;
  static Vector dx(const Vector &mu_0, const Vector &mu_1) {
    Pose3 pose_0 = fromVector(mu_0);
    Pose3 pose_1 = fromVector(mu_1);
    auto between = pose_0.between(pose_1);
    return toVector(between);
  }

  static Vector toVector(gtsam::Pose3 pose) {
    auto rot_vec = gtsam::Rot3::Logmap(pose.rotation());
    return (Vector() << rot_vec.x(), rot_vec.y(), rot_vec.z(), pose.x(),
            pose.y(), pose.z())
        .finished();
  }

  static gtsam::Pose3 fromVector(const Vector &vec) {
    return gtsam::Pose3(
        gtsam::Rot3::Expmap(
            (gtsam::Vector3() << vec(0), vec(1), vec(2)).finished()),
        gtsam::Point3(vec(3), vec(4), vec(5)));
  }
};

int main(int argc, char **argv) {
  bool robust;
  if (argc > 1) {
    robust = std::stoi(argv[1]);
  } else {
    robust = false;
  }

  Vector3 rot_gt(0.1, 0.2, 0.3);
  gtsam::Pose3 pose_gt(gtsam::Rot3::Expmap(rot_gt), gtsam::Point3(1, 2, 3));
  Eigen::Vector<double, 6> Sigma;
  Sigma << 0.05, 0.05, 0.05, 0.3, 0.3, 0.3;

  int num_samples = 100;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<std::normal_distribution<>> d_point(6);
  d_point[0] = std::normal_distribution<>(rot_gt.x(), Sigma(0));
  d_point[1] = std::normal_distribution<>(rot_gt.y(), Sigma(1));
  d_point[2] = std::normal_distribution<>(rot_gt.z(), Sigma(2));
  d_point[3] = std::normal_distribution<>(pose_gt.x(), Sigma(3));
  d_point[4] = std::normal_distribution<>(pose_gt.y(), Sigma(4));
  d_point[5] = std::normal_distribution<>(pose_gt.z(), Sigma(5));

  // generate an initial sample
  Eigen::Vector<double, 6> initial_sample;
  for (size_t j = 0; j < 6; j++) {
    initial_sample(j) = d_point[j](gen);
  }
  Variable<6>::Ptr var = std::make_shared<Variable<6>>(
      Gaussian<6>::fromMuSigma(initial_sample, Sigma.asDiagonal()));
  std::unique_ptr<Huber<6>> robust_kernel =
      robust ? std::make_unique<Huber<6>>(Sigma, Sigma * 2) : nullptr;
  Factor<6, GroupOpsPose3> factor(var, Sigma, std::move(robust_kernel));

  float outlier_ratio = 0.2;
  for (size_t i = 0; i < num_samples; i++) {
    Eigen::Vector<double, 6> sample;
    if (std::rand() % 100 < outlier_ratio * 100) {
      sample = Eigen::Vector<double, 6>::Random();
    } else {
      for (size_t j = 0; j < 6; j++) {
        sample(j) = d_point[j](gen);
      }
    }

    auto message = Gaussian<6>::fromMuSigma(sample, Sigma.asDiagonal());
    factor.update(message);
  }

  std::cout << "gt: " << pose_gt << std::endl;
  std::cout << "gt rot: " << Rot3::Logmap(pose_gt.rotation()).transpose()
            << std::endl;
  std::cout << "mu: " << var->mu().transpose() << std::endl;

  return 0;
}