#include "gbpc/factor.h"
#include "gbpc/variable.h"

template <> class gbpc::Variable<1>;
template <> class gbpc::Variable<6>;

template <int Dim> class GroupOpsDouble {
public:
  using Vector = Eigen::Vector<double, Dim>;
  static Vector dx(const Vector &mu_0, const Vector &mu_1) {
    return mu_1 - mu_0;
  }
};

template <> class gbpc::Factor<1, GroupOpsDouble<1>>;
template <> class gbpc::Factor<1, GroupOpsDouble<6>>;
