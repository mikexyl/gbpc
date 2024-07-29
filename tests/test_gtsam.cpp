#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include "gbpc/factor.h"
#include "gbpc/graph.h"
#include "gbpc/variable.h"

using namespace gbpc;
using namespace gtsam;

int main(int argc, char** argv) {
  Vector3 rot_gt(0.1, 0.2, 0.3);
  gtsam::Pose3 pose_gt(gtsam::Rot3::Expmap(rot_gt), gtsam::Point3(1, 2, 3));
  Eigen::Vector<double, 6> Sigma;
  Sigma << 0.05, 0.05, 0.05, 0.3, 0.3, 0.3;

  int num_samples = 100;

  std::mt19937 gen(0);

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
  auto initial = Belief<Pose3>(0, initial_sample, Sigma.asDiagonal(), 1);

  Graph graph;
  gbpc::Factor::shared_ptr factor =
      graph.add(std::make_shared<PriorFactor<Pose3>>(initial));

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

    auto message = Belief<Pose3>(0, sample, Sigma.asDiagonal(), 1);
    factor->update(message, GaussianMergeType::Merge);
  }

  std::cout << "gt: " << pose_gt << std::endl;
  std::cout << "gt rot: " << Rot3::Logmap(pose_gt.rotation()).transpose()
            << std::endl;
  std::cout << "mu: " << graph.getNode<Point2>(0)->mu().transpose()
            << std::endl;

  return 0;
}