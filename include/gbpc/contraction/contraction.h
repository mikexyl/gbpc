#pragma once

#include "gbpc/gaussian.h"

namespace gbpc {

class Gaussian;

struct Contraction {
  using VALUE = Pose3;

  Contraction(UpdateParams params) : params_(params) {}
  virtual Gaussian operator()(const Gaussian& curr, const Gaussian& next) = 0;

  UpdateParams params_;
};

}  // namespace gbpc