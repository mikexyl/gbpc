#ifndef GBPC_FACTOR_H_
#define GBPC_FACTOR_H_

#include <gtsam/slam/BetweenFactor.h>

#include <Eigen/Eigen>

#include "gbpc/gaussian.h"
#include "gbpc/variable.h"

namespace gbpc {

class Factor {
 public:
  using shared_ptr = std::shared_ptr<Factor>;
  using Node = Gaussian;

  Factor(const std::vector<Node::shared_ptr>& adj_vars) : adj_vars_(adj_vars) {}
  virtual ~Factor() = default;

  virtual std::string update(const Gaussian& message,
                             GaussianMergeType merge_type) = 0;

  auto const& adj_vars() { return adj_vars_; }

  KeySet keys() const {
    KeySet keys;
    for (auto& adj_var : adj_vars_) {
      keys.insert(adj_var->key());
    }
    return keys;
  }

 protected:
  std::vector<Node::shared_ptr> adj_vars_;
  Key factor_key_;
};

template <PoseConcept VALUE>
class PriorFactor : public Factor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using This = PriorFactor<VALUE>;
  using shared_ptr = std::shared_ptr<This>;

  using AdjVar = Variable<VALUE>;
  using Message = Belief<VALUE>;

  explicit PriorFactor(const AdjVar::shared_ptr& var) : Factor({var}) {}

  Variable<VALUE>* var() {
    return static_cast<Variable<VALUE>*>(adj_vars_[0].get());
  }

  std::string update(const Gaussian& message,
                     GaussianMergeType merge_type) override {
    std::stringstream ss;
    ss << "Factor::update: \n"
       << var()->mu().transpose() << " : "
       << var()->Sigma().diagonal().transpose() << " + " << std::endl
       << "  " << message.mu().transpose() << " : "
       << message.Sigma().diagonal().transpose() << " = ";

    var()->update({message}, merge_type);

    ss << var()->mu().transpose() << " : "
       << var()->Sigma().diagonal().transpose();

    return ss.str();
  }

 protected:
  Belief<VALUE> factor_belief_;
};

}  // namespace gbpc

#endif  // GBPC_FACTOR_H_
