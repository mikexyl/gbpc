#ifndef GBPC_GRAPH
#define GBPC_GRAPH

#include <gtsam/nonlinear/ISAM2Clique.h>

#include <queue>

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
  using shared_ptr = std::shared_ptr<Graph>;

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

  void optimize(Factor::shared_ptr root = nullptr) {
    if (factors_.empty()) {
      return;
    }
    if (root == nullptr) {
      root = *factors_.begin();
    }
    root->send();

    std::set<Node::shared_ptr> visited_nodes;
    std::queue<Node::shared_ptr> queue;
    queue.push(root);

    while (not queue.empty()) {
      auto node = queue.front();
      queue.pop();

      node->update();
      node->send(visited_nodes);

      visited_nodes.insert(node);

      for (auto const& neighbor : node->neighbors()) {
        if (visited_nodes.find(neighbor) == visited_nodes.end()) {
          queue.push(neighbor);
        }
      }
    }

    // clearFactorMessages();
    // clearVariableMessages();
  }

  auto const& vars() const { return vars_; }
  auto const& factors() const { return factors_; }

  GraphAndValues toGtsam() {
    NonlinearFactorGraph::shared_ptr graph(new NonlinearFactorGraph);
    Values::shared_ptr values(new Values);

    for (auto const& factor : factors_) {
      auto gtsam = factor->gtsam();
      graph->add(*gtsam.first);
      values->insert_or_assign(*gtsam.second);
    }

    return {graph, values};
  }

  void buildBayesTree() {
    auto const& [graph, values] = toGtsam();

    // BayesTree<ISAM2Clique> tree{};
  }

  Graph::shared_ptr solveByGtsam() {
    auto const& [graph, values] = toGtsam();

    LevenbergMarquardtOptimizer optimizer(*graph, *values);
    auto result = optimizer.optimize();
    Marginals marginals(*graph, result);

    std::vector<Gaussian> gaussians;
    for (auto const& [key, value] : result) {
      auto mu = traits<Point2>::Logmap(value.cast<Point2>());
      auto sigma = marginals.marginalCovariance(key);
      gaussians.emplace_back(key, mu, sigma, 1);
    }

    Graph::shared_ptr gbp_graph(new Graph);
    for (auto const& gaussian : gaussians) {
      auto var = std::make_shared<Variable<Point2>>(gaussian);
      gbp_graph->vars_.emplace(var->key(), var);
    }

    return gbp_graph;
  }

 protected:
  std::unordered_map<Key, Node::shared_ptr> vars_;
  std::set<Factor::shared_ptr> factors_;
};

}  // namespace gbpc
#endif  // GBPC_GRAPH
