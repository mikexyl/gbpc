#ifndef GBPC_GRAPH
#define GBPC_GRAPH

#include "gbpc/exceptions.h"
#include "gbpc/factor.h"
#include "gbpc/variable.h"

namespace gbpc {

using Key = size_t;

template <int Dim, class GroupOps>
class Graph : public std::map<size_t, typename Factor<Dim, GroupOps>::Ptr> {
public:
  using FactorT = Factor<Dim, GroupOps>;
  using FactorPtr = typename FactorT::Ptr;

  Graph() = default;

  void addNode(Key key, Gaussian<Dim> initial, bool robust,
               Eigen::Vector<double, Dim> Sigma = {}) {
    if (this->find(key) != this->end()) {
      throw NodeAlreadyExistsException(key);
    }

    auto var = std::make_shared<Variable<Dim>>(initial);
    std::unique_ptr<Huber<Dim>> robust_kernel =
        robust ? std::make_unique<Huber<Dim>>(Sigma, Sigma * 2) : nullptr;
    this->emplace(key, FactorPtr(new FactorT(var, std::move(robust_kernel))));
  }

  std::string
  sendMessage(Key key, Gaussian<Dim> message,
              GaussianMergeType merge_type = GaussianMergeType::Merge) {
    if (this->find(key) == this->end()) {
      throw NodeNotFoundException(key);
    }

    return this->at(key)->update(message, merge_type);
  }

  auto getNode(Key key) {
    if (this->find(key) == this->end()) {
      throw NodeNotFoundException(key);
    }

    return this->at(key)->adj_var();
  }
};

/**
 * @brief Coverage & belief graph
 *
 * @tparam Dim
 * @tparam GroupOps
 */
template <int Dim, class GroupOps> class CBGraph {
public:
  using BeliefGraphT = Graph<Dim, GroupOps>;
  using GaussianBelief = Gaussian<Dim>;
  using CoverageGraphT = Graph<Dim, GroupOps>;
  using GaussianCoverage = Gaussian<Dim>;

  CBGraph() {
    belief_graph_ = std::make_shared<BeliefGraphT>();
    coverage_graph_ = std::make_shared<CoverageGraphT>();
  }

  void addNode(Key key, GaussianBelief initial, GaussianCoverage coverage,
               bool robust, Eigen::Vector<double, Dim> Sigma = {}) {
    belief_graph_->addNode(key, initial, robust, Sigma);
    coverage_graph_->addNode(key, coverage, false);
  }

  std::string sendMessage(Key key, GaussianBelief message,
                          GaussianCoverage coverage) {
    std::stringstream ss;

    double hellinger =
        coverage_graph_->getNode(key)->gaussian().hellingerDistance(coverage);

    ss << "Hellinger: " << hellinger << std::endl;

    static constexpr float kOverlapEpsilon =
        std::numeric_limits<double>::epsilon();
    if (hellinger <= kOverlapEpsilon) {
      ss << "Overlap is too high, discard message" << std::endl;
      return ss.str();
    }

    // othewise, accept the message with relaxation
    message.relax(hellinger);

    ss << coverage_graph_->sendMessage(key, coverage,
                                       GaussianMergeType::Mixture)
       << std::endl;

    ss << belief_graph_->sendMessage(key, message, GaussianMergeType::Merge)
       << std::endl;

    return ss.str();
  }

  auto getNodeBelief(Key key) { return belief_graph_->getNode(key); }

  auto getNodeCoverage(Key key) { return coverage_graph_->getNode(key); }

private:
  std::shared_ptr<BeliefGraphT> belief_graph_;
  std::shared_ptr<CoverageGraphT> coverage_graph_;
};

} // namespace gbpc
#endif // GBPC_GRAPH