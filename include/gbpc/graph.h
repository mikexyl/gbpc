#ifndef GBPC_GRAPH
#define GBPC_GRAPH

#include "gbpc/exceptions.h"
#include "gbpc/factor.h"
#include "gbpc/variable.h"

namespace gbpc {

using Key = size_t;

template <int Dim, class GroupOps>
class Graph : public std::map<size_t, typename Factor<6, GroupOps>::Ptr> {
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
    std::unique_ptr<Huber<6>> robust_kernel =
        robust ? std::make_unique<Huber<6>>(Sigma, Sigma * 2) : nullptr;
    this->emplace(key, FactorPtr(new FactorT(var, std::move(robust_kernel))));
  }

  void sendMessage(Key key, Gaussian<Dim> message) {
    if (this->find(key) == this->end()) {
      throw NodeNotFoundException(key);
    }

    this->at(key)->update(message);
  }

  auto getNode(Key key) {
    if (this->find(key) == this->end()) {
      throw NodeNotFoundException(key);
    }

    return this->at(key)->adj_var();
  }
};
} // namespace gbpc
#endif // GBPC_GRAPH