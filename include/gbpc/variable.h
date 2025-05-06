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

  template <typename... Args>
  Variable(Args&&... args) : Base(std::forward<Args>(args)...) {}

  void setBelief(const Base& belief) { Base::replace(belief); }
};

}  // namespace gbpc

#endif  // GBPC_VARIABLE_NODE_H_
