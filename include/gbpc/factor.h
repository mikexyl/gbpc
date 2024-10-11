#ifndef GBPC_FACTOR_H_
#define GBPC_FACTOR_H_

#include <gtsam/slam/dataset.h>

#include <Eigen/Eigen>
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

  virtual gtsam::GraphAndValues gtsam() = 0;

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
  using BeliefT = Belief<VALUE>;
  using Message = BeliefT;

  explicit BetweenFactor(const Message& measured) : Factor(measured) {}

  Variable<VALUE>* var1() {
    return static_cast<Variable<VALUE>*>(adj_vars()[0].get());
  }

  Variable<VALUE>* var2() {
    return static_cast<Variable<VALUE>*>(adj_vars()[1].get());
  }

  std::optional<Gaussian> potential(const Node::shared_ptr& var) override {
    assert(var == adj_vars()[0] || var == adj_vars()[1]);

    if (var == adj_vars()[0]) {
      Gaussian message(*this);
      message.merge(*adj_vars()[0], false);
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

  gtsam::GraphAndValues gtsam() override {
    NonlinearFactorGraph::shared_ptr graph(new NonlinearFactorGraph());
    auto g0 = traits<VALUE>::Expmap(this->mu());
    auto noise = gtsam::noiseModel::Gaussian::Covariance(this->Sigma());

    typename gtsam::BetweenFactor<VALUE>::shared_ptr factor(
        new gtsam::BetweenFactor<VALUE>(
            var1()->key(), var2()->key(), g0, noise));
    graph->push_back(factor);

    Values::shared_ptr values(new Values());
    auto value1 = traits<VALUE>::Expmap(var1()->mu());
    auto value2 = traits<VALUE>::Expmap(var2()->mu());
    values->insert(var1()->key(), value1);
    values->insert(var2()->key(), value2);

    return {graph, values};
  }
};

template <class VALUE>
class PriorFactor : public Factor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using This = PriorFactor<VALUE>;
  using shared_ptr = std::shared_ptr<This>;

  using AdjVar = Variable<VALUE>;
  using BeliefT = Belief<VALUE>;
  using Message = BeliefT;

  explicit PriorFactor(const BeliefT& prior)
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

  UpdateResult update(const Gaussian& message, UpdateParams update_params) {
    std::stringstream ss;
    ss << "Factor::update: \n"
       << var()->mu().transpose() << " : "
       << var()->Sigma().diagonal().transpose() << " + " << std::endl
       << "  " << message.mu().transpose() << " : "
       << message.Sigma().diagonal().transpose() << " = ";

    UpdateResult results;
    varAsGaussian()->update({message}, update_params, &results);

    ss << var()->mu().transpose() << " : "
       << var()->Sigma().diagonal().transpose();

    results.message = ss.str();
    return results;
  }

  gtsam::GraphAndValues gtsam() override {
    NonlinearFactorGraph::shared_ptr graph(new NonlinearFactorGraph());

    auto g0 = traits<VALUE>::Expmap(this->mu());
    auto noise = gtsam::noiseModel::Gaussian::Covariance(this->Sigma());
    typename gtsam::PriorFactor<VALUE>::shared_ptr prior(
        new gtsam::PriorFactor<VALUE>(var()->key(), g0, noise));
    graph->push_back(prior);

    Values::shared_ptr values(new Values());
    auto value = traits<VALUE>::Expmap(this->mu());
    values->insert(var()->key(), value);

    return {graph, values};
  }
};

}  // namespace gbpc

#endif  // GBPC_FACTOR_H_
