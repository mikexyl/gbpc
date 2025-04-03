#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

#include "gbpc/gbpc.h"

using namespace gtsam;
using namespace gbpc;

int main() {
  using namespace gtsam;

  // Setup keys
  Key x_key = Symbol('x', 0);
  Key y_key = Symbol('y', 0);

  // Linearization points
  Pose3 x_lin = Pose3(Rot3::Identity(), Point3(0, 0, 0));
  Pose3 y_lin = Pose3(Rot3::Identity(), Point3(1, 0, 0));  // y is 1m ahead of x

  // PERTURB x slightly: 5 cm in x, 2 cm in y, 1 deg yaw
  Vector6 dx;
  dx << 0.1, 0.1, 0.1, 0.5, 0.2, 0.0;  // [roll pitch yaw x y z]
  Pose3 x_perturbed = x_lin.retract(dx);

  // Create noisy measurement: z = x_perturbed^-1 * y_lin
  Pose3 measurement = x_lin.between(y_lin);

  x_lin = x_perturbed;

  // Correct noise model [rotation first!]
  auto model = noiseModel::Diagonal::Sigmas(
      (Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());

  // Factor between x and y
  auto factor = gtsam::BetweenFactor<Pose3>(x_key, y_key, measurement, model);

  // Belief: assume Gaussian on x centered at x_lin
  Matrix Sigma_x = Matrix::Identity(6, 6) * 0.02;  // moderate uncertainty

  Values values;
  values.insert(x_key, x_lin);
  values.insert(y_key, y_lin);

  gbpc::Nodes vars;
  vars[x_key] = std::make_shared<gbpc::Belief<Pose3>>(
      x_key, Pose3::Logmap(x_lin), Sigma_x, 1);
  vars[y_key] = std::make_shared<gbpc::Belief<Pose3>>(
      y_key, Pose3::Logmap(y_lin), Matrix::Identity(6, 6), 1);

  // Compute marginal covariance of y via explicit Schur complement
  gbpc::Factor::updateFactorToVar(y_key, factor, &vars, true);

  std::cout << vars.print() << std::endl;

  // create a factor graph and use Marginals to compute the marginal
  // covariance
  NonlinearFactorGraph graph;
  graph.add(factor);
  graph.addPrior(
      x_key,
      x_lin,
      noiseModel::Gaussian::Covariance(Sigma_x, true));  // add prior on x

  // optimize with GN
  GaussNewtonOptimizer optimizer(graph, values);
  auto result = optimizer.optimize();
  result.print("Result:\n");
  // print mu of x and y
  std::cout << "GN: mu of x:\n"
            << Pose3::Logmap(result.at<Pose3>(x_key)).transpose() << std::endl;
  std::cout << "GN: mu of y:\n"
            << Pose3::Logmap(result.at<Pose3>(y_key)).transpose() << std::endl;

  Marginals marginals(graph, result);
  Matrix Sigma_y_marginal = marginals.marginalCovariance(y_key);
  std::cout << "Estimated marginal covariance of y (marginals):\n"
            << Sigma_y_marginal << std::endl;
  Matrix Sigma_x_marginal = marginals.marginalCovariance(x_key);
  std::cout << "Estimated marginal covariance of x (marginals):\n"
            << Sigma_x_marginal << std::endl;

  return 0;
}
