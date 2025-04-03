#ifndef GBPC_FACTOR_H_
#define GBPC_FACTOR_H_

#include <gtsam/slam/dataset.h>

#include <boost/concept_check.hpp>
#include <memory>

#include "gbpc/gaussian.h"
#include "gbpc/variable.h"

namespace gbpc {

class Factor : public Node {
 public:
  using shared_ptr = std::shared_ptr<Factor>;

  Factor(const Gaussian& measured) : Node(measured) {}
  virtual ~Factor() = default;

  Factor::shared_ptr getSharedFactor() {
    // This will throw an error if the object is not managed by a
    // std::shared_ptr
    return std::dynamic_pointer_cast<Factor>(this->shared_from_this());
  }

  auto const& adj_vars() const { return neighbors(); }
  void addAdjVar(const Node::shared_ptr& adj_var) {
    addNeighbor(adj_var);
    adj_var->addNeighbor(this->getSharedFactor());
  }
  void addAdjVar(const std::vector<Node::shared_ptr>& adj_vars) {
    for (auto adj_var : adj_vars) {
      addAdjVar(adj_var);
    }
  }

  KeySet keys() const {
    KeySet keys;
    for (auto& adj_var : this->adj_vars()) {
      keys.insert(adj_var->key());
    }
    return keys;
  }

  virtual void update(std::vector<Gaussian>, UpdateParams, UpdateResult*) {
    throw "Factor::update not implemented";
  };

  auto gtsam() { return gtsam_factor_; }

 protected:
  NonlinearFactor::shared_ptr gtsam_factor_;

 public:
  static void updateFactorToVar(
      Key target_key,
      const gtsam::NoiseModelFactor& factor,
      std::unordered_map<gtsam::Key, std::shared_ptr<gbpc::Node>>* vars,
      bool update_mu = false,
      const UpdateParams& params = UpdateParams(),
      UpdateResult* result = nullptr) {
    // Step 1: Linearize factor and cast to JacobianFactor
    Values values;
    VectorValues vec_values;
    for (auto key : factor.keys()) {
      auto it = vars->find(key);
      if (it == vars->end()) throw std::runtime_error("Key not found in vars_");
      (*it).second->addToValues(&values);
      vec_values.insert(key, (*it).second->mu());
    }
    auto lin_f = factor.linearize(values);
    auto jac = boost::dynamic_pointer_cast<JacobianFactor>(lin_f);
    if (!jac)
      throw std::runtime_error("Expected JacobianFactor from linearization");

    // Only one variable in the factor — must be the target
    if (jac->keys().size() == 1) {
      if (jac->keys()[0] == target_key) {
        // It's a unary factor on the target key → return its covariance
        const auto& model = jac->get_model();  // Gaussian noise model

        if (!model)
          throw std::runtime_error("Unary prior factor has no noise model");

        // The information matrix is RᵀR (from QR)
        const Matrix lambda = model->information();  // = RᵀR
        std::cout << "Updating prior factor" << std::endl;
        Vector new_eta = vars->at(target_key)->eta();
        if (update_mu) {
          new_eta = jac->getb();  // natural vector for target
        }
        vars->at(target_key)
            ->updateFromCanonical(
                new_eta,
                lambda,
                params,
                result);  // update target with new mu and lambda
        return;
      } else {
        throw std::runtime_error(
            "Factor has only one variable, but it's not the target key");
      }
    }

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
      auto node_it = vars->find(k);
      if (node_it == vars->end())
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
      perm_indices.push_back(col_start - target_dim +
                             i);  // target block at end

    P.indices() =
        Eigen::VectorXi::Map(perm_indices.data(), perm_indices.size());

    Matrix Lambda_reordered = P.transpose() * Lambda_total * P;

    // Partition into blocks
    Matrix Lambda_rr = Lambda_reordered.topLeftCorner(rest_dim, rest_dim);
    Matrix Lambda_rt = Lambda_reordered.topRightCorner(rest_dim, target_dim);
    Matrix Lambda_tr = Lambda_reordered.bottomLeftCorner(target_dim, rest_dim);
    Matrix Lambda_tt =
        Lambda_reordered.bottomRightCorner(target_dim, target_dim);

    // Step 6: Schur complement
    Matrix Lambda_tt_marginal =
        Lambda_tt - Lambda_tr * Lambda_rr.inverse() * Lambda_rt;
    Vector new_eta = vars->at(target_key)->eta();

    // Step 7: Update mu if needed
    if (update_mu) {
      Vector b = jac->getb();
      Vector r = factor.whitenedError(values);
      std::cout << "r: " << r.transpose() << std::endl;
      Vector eta = -H.transpose() * r;
      Vector eta_reordered = P.transpose() * eta;
      Vector eta_r = eta_reordered.head(rest_dim);    // natural vec for rest
      Vector eta_t = eta_reordered.tail(target_dim);  // natural vec for target
      Vector delta_eta = eta_t - Lambda_tr * Lambda_rr.inverse() * eta_r;
      new_eta = 0.1 * delta_eta + vars->at(target_key)->eta();
      std::cout << "delta eta: " << delta_eta.transpose() << std::endl;
      std::cout << "new eta: " << new_eta.transpose() << std::endl;
    }

    std::cout << "new sigma: " << Lambda_tt_marginal.inverse() << std::endl;
    std::cout << "new mu"
              << (Lambda_tt_marginal.inverse() * new_eta).transpose()
              << std::endl;

    // Step 7: Update Lambda
    vars->at(target_key)
        ->updateFromCanonical(new_eta,
                              Lambda_tt_marginal,
                              params,
                              result);  // update target with new mu and lambda
  }
};

}  // namespace gbpc

#endif  // GBPC_FACTOR_H_
