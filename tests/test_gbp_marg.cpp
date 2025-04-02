#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

#include "gbpc/gbpc.h"

using namespace gtsam;
using namespace gbpc;

// Assumes each Node in vars_ provides: Key key(), Matrix Lambda(), Vector mu()
Matrix computeMarginalCovariance(
    Key target_key,
    const gtsam::NonlinearFactor& factor,
    const std::unordered_map<gtsam::Key, std::shared_ptr<gbpc::Node>>& vars,
    const gtsam::Values& values) {
  // Step 1: Linearize factor and cast to JacobianFactor
  auto lin_f = factor.linearize(values);
  auto jac = boost::dynamic_pointer_cast<JacobianFactor>(lin_f);
  if (!jac)
    throw std::runtime_error("Expected JacobianFactor from linearization");

  // Step 2: Gather keys and ensure target is present
  const auto& keys = jac->keys();
  const size_t num_vars = keys.size();
  auto it_target = std::find(keys.begin(), keys.end(), target_key);
  if (it_target == keys.end())
    throw std::runtime_error("Target key not involved in the factor");

  const size_t target_index = std::distance(keys.begin(), it_target);

  // Step 3: Build joint Jacobian matrix H and prior info matrix Lambda_prior
  std::vector<int> dims;
  std::vector<Matrix> blocks;
  std::vector<Matrix> priors;

  size_t total_dim = 0;
  for (size_t i = 0; i < num_vars; ++i) {
    Key k = keys[i];
    auto node_it = vars.find(k);
    if (node_it == vars.end())
      throw std::runtime_error("Missing prior for key in vars_");

    Matrix Hi = jac->getA(jac->keys().begin() + i);

    Matrix Lambda_i = node_it->second->Lambda();
    if (k == target_key) {
      Lambda_i = Matrix::Zero(Hi.cols(), Hi.cols());  // ← no prior on target
    } else {
      Lambda_i = node_it->second->Lambda();
    }

    dims.push_back(Hi.cols());
    blocks.push_back(Hi);
    priors.push_back(Lambda_i);
    total_dim += Hi.cols();
  }

  // Construct H (concatenated Jacobians) and Lambda_prior (block diagonal)
  Matrix H(jac->rows(), total_dim);
  Matrix Lambda_prior = Matrix::Zero(total_dim, total_dim);

  size_t col_start = 0;
  for (size_t i = 0; i < num_vars; ++i) {
    int dim = dims[i];

    H.middleCols(col_start, dim) = blocks[i];
    Lambda_prior.block(col_start, col_start, dim, dim) = priors[i];

    col_start += dim;
  }

  // Step 4: Compute total information matrix
  Matrix Lambda_total = Lambda_prior + H.transpose() * H;

  // Step 5: Partition into (target, rest)
  // Reorder so that target block is last
  std::vector<int> ordering;  // new column/row ordering
  int target_dim = dims[target_index];
  int rest_dim = total_dim - target_dim;

  // map: original_index → reordered_index
  std::vector<std::pair<int, int>> reorder_pairs;
  col_start = 0;
  for (size_t i = 0; i < num_vars; ++i) {
    int dim = dims[i];
    if (i != target_index) {
      reorder_pairs.emplace_back(col_start, dim);
    }
    col_start += dim;
  }
  int target_offset = 0;
  for (size_t i = 0; i < reorder_pairs.size(); ++i)
    target_offset += reorder_pairs[i].second;

  // Build permutation matrix
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(total_dim);
  std::vector<int> perm_indices;

  for (const auto& [offset, dim] : reorder_pairs)
    for (int i = 0; i < dim; ++i) perm_indices.push_back(offset + i);

  for (int i = 0; i < target_dim; ++i)
    perm_indices.push_back(col_start - target_dim + i);  // target block at end

  P.indices() = Eigen::VectorXi::Map(perm_indices.data(), perm_indices.size());

  Matrix Lambda_reordered = P.transpose() * Lambda_total * P;

  // Partition into blocks
  Matrix Lambda_rr = Lambda_reordered.topLeftCorner(rest_dim, rest_dim);
  Matrix Lambda_rt = Lambda_reordered.topRightCorner(rest_dim, target_dim);
  Matrix Lambda_tr = Lambda_reordered.bottomLeftCorner(target_dim, rest_dim);
  Matrix Lambda_tt = Lambda_reordered.bottomRightCorner(target_dim, target_dim);

  // Step 6: Schur complement
  Matrix Lambda_tt_marginal =
      Lambda_tt - Lambda_tr * Lambda_rr.inverse() * Lambda_rt;

  // Step 7: Invert to get marginal covariance of target
  return Lambda_tt_marginal.inverse();
}

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

  std::unordered_map<Key, std::shared_ptr<gbpc::Node>> vars;
  vars[x_key] = std::make_shared<gbpc::Belief<Pose3>>(
      x_key, Pose3::Logmap(x_lin), Sigma_x, 1);
  vars[y_key] = std::make_shared<gbpc::Belief<Pose3>>(
      y_key, Pose3::Logmap(y_lin), Matrix::Identity(6, 6), 1);

  // Compute marginal covariance of y via explicit Schur complement
  Matrix Sigma_y = computeMarginalCovariance(y_key, factor, vars, values);

  std::cout << "mu_x: " << Pose3::Logmap(x_lin).transpose() << std::endl;
  std::cout << "x's covariance:\n" << Sigma_x << std::endl;
  std::cout << "factor's measurement:\n"
            << Pose3::Logmap(measurement).transpose() << std::endl;
  std::cout << "factor's noise model sqrt_info:\n" << model->R() << std::endl;
  std::cout << "residual: " << factor.error(values) << std::endl;
  std::cout << "=== Schur Complement Test with Perturbed Input ===\n";
  std::cout << "Estimated marginal covariance of y:\n" << Sigma_y << std::endl;

  // create a factor graph and use Marginals to compute the marginal covariance
  NonlinearFactorGraph graph;
  graph.add(factor);
  graph.addPrior(
      x_key,
      x_lin,
      noiseModel::Gaussian::Covariance(Sigma_x, true));  // add prior on x

  Marginals marginals(graph, values);
  Matrix Sigma_y_marginal = marginals.marginalCovariance(y_key);
  std::cout << "Estimated marginal covariance of y (marginals):\n"
            << Sigma_y_marginal << std::endl;
  Matrix Sigma_x_marginal = marginals.marginalCovariance(x_key);
  std::cout << "Estimated marginal covariance of x (marginals):\n"
            << Sigma_x_marginal << std::endl;

  return 0;
}
