#ifndef GBPC_GRAPH
#define GBPC_GRAPH

#include "gbpc/exceptions.h"
#include "gbpc/factor.h"
#include "gbpc/gaussian.h"
#include "gbpc/variable.h"

using namespace gtsam;

namespace gbpc {

struct GaussianKeyCompare {
  bool operator()(const Gaussian::shared_ptr& lhs,
                  const Gaussian::shared_ptr& rhs) const {
    return lhs->key() < rhs->key();
  }
};

class Graph {
 public:
  Graph() = default;

  auto add(const Factor::shared_ptr& factor) {
    for (auto const& var : factor->adj_vars()) {
      vars_.emplace(var->key(), var);
    }

    factors_.insert(factor);

    return factor;
  }

  auto add(const std::vector<Factor::shared_ptr>& factors) {
    for (auto const& factor : factors) {
      add(factor);
    }
  }

  template <typename T = Gaussian>
  auto getVar(Key key) const {
    if (vars_.find(key) != vars_.end()) {
      return std::dynamic_pointer_cast<T>(vars_.at(key));
    }

    throw NodeNotFoundException(key);
  }

  template <typename T>
  T::shared_ptr getFactor(Key key) const {
    for (auto const& factor : factors_) {
      if (factor->key() == key) {
        return std::dynamic_pointer_cast<T>(factor);
      }
    }

    throw NodeNotFoundException(key);
  }

  std::string print() {
    std::stringstream ss;
    ss << "Graph:" << std::endl;
    ss << "Variables:" << std::endl;
    for (auto const& [_, var] : vars_) {
      ss << var->print() << std::endl;
    }

    ss << "Factors:" << std::endl;
    for (auto const& factor : factors_) {
      ss << factor->print() << std::endl;
    }

    return ss.str();
  }

  void clearFactorMessages() {
    for (auto const& factor : factors_) {
      factor->clearMessages();
    }
  }

  void clearVariableMessages() {
    for (auto const& [_, var] : vars_) {
      var->clearMessages();
    }
  }

  void optimize() {
    for (auto const& factor : factors_) {
      factor->send();
    }

    for (auto const& [_, var] : vars_) {
      var->send();
    }

    for (auto const& [_, var] : vars_) {
      var->update();
    }

    clearFactorMessages();
    clearVariableMessages();
  }

  bool contains(Key key) const { return vars_.find(key) != vars_.end(); }

  auto const& vars() const { return vars_; }
  auto const& factors() const { return factors_; }
  std::vector<Gaussian> solveByGtsam() {
    NonlinearFactorGraph graph;
    Values values;

    for (auto const& factor : factors_) {
      auto gtsam = factor->gtsam();
      graph.add(*gtsam.first);
      values.insert_or_assign(*gtsam.second);
    }

    LevenbergMarquardtOptimizer optimizer(graph, values);
    auto result = optimizer.optimize();
    Marginals marginals(graph, result);

    std::vector<Gaussian> gaussians;
    for (auto const& [key, value] : result) {
      auto mu = traits<Point2>::Logmap(value.cast<Point2>());
      auto sigma = marginals.marginalCovariance(key);
      gaussians.emplace_back(key, mu, sigma, 1);
    }

    return gaussians;
  }

 protected:
  std::unordered_map<Key, Node::shared_ptr> vars_;
  std::set<Factor::shared_ptr> factors_;
};

}  // namespace gbpc
#endif  // GBPC_GRAPH
