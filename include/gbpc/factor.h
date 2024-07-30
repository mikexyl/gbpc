#ifndef GBPC_FACTOR_H_
#define GBPC_FACTOR_H_

#include <Eigen/Eigen>

#include "gbpc/gaussian.h"
#include "gbpc/variable.h"

namespace gbpc {

class Factor : public Node {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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

 protected:
  Key factor_key_;
};

template <class VALUE>
class BetweenFactor : public Factor {
  // Check that VALUE type is a testable Lie group
  BOOST_CONCEPT_ASSERT((IsTestable<VALUE>));
  BOOST_CONCEPT_ASSERT((IsLieGroup<VALUE>));

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using This = BetweenFactor<VALUE>;
  using shared_ptr = std::shared_ptr<This>;
  using AdjVar = Variable<VALUE>;
  using Belief = Belief<VALUE>;
  using Message = Belief;

  explicit BetweenFactor(const Message& measured) : Factor(measured) {}

  Variable<VALUE>* var1() {
    return static_cast<Variable<VALUE>*>(adj_vars()[0].get());
  }

  Variable<VALUE>* var2() {
    return static_cast<Variable<VALUE>*>(adj_vars()[1].get());
  }

  VALUE constructFromVector(const Eigen::VectorXd& vec) {
    throw "Not implemented";
  }

  std::optional<Gaussian> potential(const Node::shared_ptr& var) {
    assert(var == adj_vars()[0] || var == adj_vars()[1]);

    if (var == adj_vars()[0]) {
      Gaussian message(*this);
      message.merge(*adj_vars()[1], false);
      return message;
    } else if (var == adj_vars()[1]) {
      auto measured = traits<VALUE>::Expmap(this->mu());
      auto inverse = traits<VALUE>::Inverse(measured);
      auto inverse_mu = traits<VALUE>::Logmap(inverse);
      Gaussian inverse_message(
          this->key(), inverse_mu, this->Sigma(), this->N());
      inverse_message.merge(*adj_vars()[1], false);
      return inverse_message;
    }

    return std::nullopt;
  }
};

template <class VALUE>
class PriorFactor : public Factor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using This = PriorFactor<VALUE>;
  using shared_ptr = std::shared_ptr<This>;

  using AdjVar = Variable<VALUE>;
  using Belief = Belief<VALUE>;
  using Message = Belief;

  explicit PriorFactor(const Belief& prior)
      : Factor(static_cast<Gaussian>(prior)) {}

  Variable<VALUE>* var() {
    return static_cast<Variable<VALUE>*>(adj_vars()[0].get());
  }

  Gaussian* varAsGaussian() {
    return static_cast<Gaussian*>(adj_vars()[0].get());
  }

  std::optional<Gaussian> potential(const Node::shared_ptr& var) override {
    throw "should never be called";
  }

  Gaussian prior() const override { return static_cast<Gaussian>(*this); }

  std::string update(const Gaussian& message, GaussianMergeType merge_type) {
    std::stringstream ss;
    ss << "Factor::update: \n"
       << var()->mu().transpose() << " : "
       << var()->Sigma().diagonal().transpose() << " + " << std::endl
       << "  " << message.mu().transpose() << " : "
       << message.Sigma().diagonal().transpose() << " = ";

    varAsGaussian()->update({message}, merge_type);

    ss << var()->mu().transpose() << " : "
       << var()->Sigma().diagonal().transpose();

    return ss.str();
  }
};

}  // namespace gbpc

#endif  // GBPC_FACTOR_H_
