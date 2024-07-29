#include "gbpc/gaussian.h"

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>

using namespace gtsam;

namespace gbpc {
template <>
class Belief<Pose2>;

template <>
class Belief<Point2>;

template <>
class Belief<Pose3>;

}  // namespace gbpc