#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include "gbpc/variable_node.h"

using namespace gbpc;

int main() {
  gtsam::Point3 point_gt(1, 2, 3);
  Eigen::Vector3d Sigma = Eigen::Vector3d::Ones();

  int num_samples = 100;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<std::normal_distribution<>> d_point(3);
  d_point[0] = std::normal_distribution<>(point_gt.x(), Sigma(0));
  d_point[1] = std::normal_distribution<>(point_gt.y(), Sigma(1));
  d_point[2] = std::normal_distribution<>(point_gt.z(), Sigma(2));

  VariableNode<3> vn_point;
  std::vector<Gaussian<3>> messages;
  for (size_t i = 0; i < num_samples; i++) {
    gtsam::Point3 point(d_point[0](gen), d_point[1](gen), d_point[2](gen));
    auto message = Gaussian<3>::fromMuSigma(point, Sigma.asDiagonal());
    messages.push_back(message);
  }

  vn_point.update(messages);

  std::cout << "point_gt: " << point_gt << std::endl;
  std::cout << "point_mu: " << vn_point.mu() << std::endl;

  return 0;
}