#ifndef GBPC_VARIABLE_NODE_H_
#define GBPC_VARIABLE_NODE_H_

#include <Eigen/Eigen>
#include <memory>
#include <optional>

#include "gaussian.h"

namespace gbpc {

template <typename VALUE>
class Variable : public Belief<VALUE> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Base = Belief<VALUE>;
  using This = Variable<VALUE>;
  using shared_ptr = std::shared_ptr<This>;

  Variable(const Base& initial) : Base(initial) {}

  void setBelief(const Base& belief) { Base::replace(belief); }
};

}  // namespace gbpc

#endif  // GBPC_VARIABLE_NODE_H_
