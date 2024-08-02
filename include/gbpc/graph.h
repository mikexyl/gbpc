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

class GBPClique : public std::vector<Node::shared_ptr> {
 public:
  using shared_ptr = std::shared_ptr<GBPClique>;

  GBPClique(bool is_root = false) : is_root_(is_root) {}
  virtual ~GBPClique() = default;

  std::string print() {
    std::stringstream ss;
    ss << "Clique: ";
    for (auto const& node : *this) {
      ss << DefaultKeyFormatter(node->key()) << " ";
    }

    if (is_root_) {
      ss << " (root)";
    }
    return ss.str();
  }

 private:
  bool is_root_{false};
};

struct GBPUpdateParams {
  Node::shared_ptr root{nullptr};
  Node::NodePairFunc pre_pass{};
  Node::NodePairFunc post_pass{};
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

  void remove(const Factor::shared_ptr& factor) { factors_.erase(factor); }

  void remove(const std::vector<Factor::shared_ptr>& factors) {
    for (auto const& factor : factors) {
      remove(factor);
    }
  }

  auto add(const std::vector<Factor::shared_ptr>& factors) {
    for (auto const& factor : factors) {
      add(factor);
    }
  }

  bool has(const Factor::shared_ptr& factor) {
    return factors_.find(factor) != factors_.end();
  }

  bool hasAny(const std::vector<Factor::shared_ptr>& factors) {
    for (auto const& factor : factors) {
      if (has(factor)) {
        return true;
      }
    }

    return false;
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

  void optimize(const GBPUpdateParams& params = {}) {
    auto root = params.root;
    if (factors_.empty()) {
      return;
    }
    if (root == nullptr) {
      root = *factors_.begin();
    }
    root->send(params.post_pass);

    std::set<Node::shared_ptr> visited_vars;
    std::set<Node::shared_ptr> visited_factors;
    std::queue<Node::shared_ptr> queue;
    queue.push(root);

    while (not queue.empty()) {
      auto node = queue.front();
      queue.pop();

      node->update();
      node->send(params.post_pass, visited_factors);

      if (auto as_factor = std::dynamic_pointer_cast<Factor>(node)) {
        visited_factors.insert(as_factor);
      }

      visited_vars.insert(node);

      for (auto const& neighbor : node->neighbors()) {
        if (visited_vars.find(neighbor) == visited_vars.end()) {
          queue.push(neighbor);
        }
      }
    }

    clearFactorMessages();
    clearVariableMessages();
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

  auto buildBayesTree() {
    auto const& [graph, values] = toGtsam();

    // Step 4: Linearize the graph around the result to get a
    // GaussianFactorGraph
    GaussianFactorGraph::shared_ptr linearGraph = graph->linearize(*values);

    // Step 5: Create an ordering for variable elimination
    Ordering ordering = Ordering::Colamd(*linearGraph);

    // Step 6: Eliminate the factor graph to obtain a Bayes Net
    GaussianBayesNet::shared_ptr bayesNet =
        linearGraph->eliminateSequential(ordering);

    // Step 7: Convert the Bayes Net to a Bayes Tree
    GaussianBayesTree::shared_ptr bayesTree =
        linearGraph->eliminateMultifrontal(ordering);

    cliques_.clear();
    for (auto const& [clique_key, clique] : bayesTree->nodes()) {
      cliques_.insert(fromClique(clique));
    }

    return bayesTree;
  }

  GBPClique::shared_ptr fromClique(
      const GaussianBayesTreeClique::shared_ptr& clique) {
    GBPClique::shared_ptr gbp_clique(new GBPClique(clique->isRoot()));
    for (auto const& key : clique->conditional()->keys()) {
      gbp_clique->emplace_back(vars_[key]);
    }
    return gbp_clique;
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

  auto const& cliques() const { return cliques_; }

  void optimizeRoots() {}

 protected:
  std::unordered_map<Key, Node::shared_ptr> vars_;
  std::set<Factor::shared_ptr> factors_;
  std::set<GBPClique::shared_ptr> cliques_;
};

}  // namespace gbpc
#endif  // GBPC_GRAPH
