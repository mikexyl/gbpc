#ifndef GBPC_VARIABLE_NODE_H_
#define GBPC_VARIABLE_NODE_H_

#include <Eigen/Eigen>
#include <memory>
#include <optional>

#include "gaussian.h"

namespace gbpc {

template <PoseConcept VALUE>
class Variable : public Belief<VALUE> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using This = Variable<VALUE>;
  using shared_ptr = std::shared_ptr<This>;
  using Belief = Belief<VALUE>;

  Variable(const Belief& initial) : Belief(initial) {}
};

}  // namespace gbpc

#endif  // GBPC_VARIABLE_NODE_H_
