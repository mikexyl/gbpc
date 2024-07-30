#ifndef GBPC_GRAPH
#define GBPC_GRAPH

#include "gbpc/exceptions.h"
#include "gbpc/factor.h"
#include "gbpc/gaussian.h"
#include "gbpc/variable.h"

using namespace gtsam;

namespace gbpc {

using Key = size_t;

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

  template <typename T>
  Belief<T>* getNode(Key key) {
    if (vars_.find(key) != vars_.end()) {
      return static_cast<Variable<T>*>(vars_[key].get());
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

  }

  auto const& vars() const { return vars_; }
  auto const& factors() const { return factors_; }

 protected:
  std::unordered_map<Key, Node::shared_ptr> vars_;
  std::set<Factor::shared_ptr> factors_;
};

}  // namespace gbpc
#endif  // GBPC_GRAPH
