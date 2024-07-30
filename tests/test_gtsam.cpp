#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include "gbpc/factor.h"
#include "gbpc/gaussian.h"
#include "gbpc/graph.h"
#include "gbpc/variable.h"

int main(int argc, char** argv) {
  gbpc::Graph graph;

  int num_nodes = 6;
  std::vector<gbpc::Belief<Point2>> init;
  for (int i = 0; i < num_nodes; i++) {
    init.emplace_back(gbpc::Belief<Point2>(i));
  }

  // Add variables
  std::vector<gbpc::Variable<Point2>::shared_ptr> variables;
  for (int i = 0; i < num_nodes; i++) {
    variables.push_back(std::make_shared<gbpc::Variable<Point2>>(init[i]));
  }

  for (int i = 0; i < num_nodes - 1; i++) {
    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();
    cov << 1, 0, 0, 1;
    gbpc::Belief<Point2> measured(i, Point2(50, 50), cov, 1);
    auto factor = std::make_shared<gbpc::BetweenFactor<Point2>>(measured);
    factor->addAdjVar({variables[i], variables[i + 1]});
    graph.add(factor);
  }

  auto prior_factor = std::make_shared<gbpc::PriorFactor<Point2>>(
      gbpc::Belief<Point2>(0, Point2(0, 0), Eigen::Matrix2d::Identity(), 1));
  prior_factor->addAdjVar(variables[0]);
  graph.add(prior_factor);
}
