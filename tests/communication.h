#pragma once

#include "nanoflann.hpp"
#include "robot.h"

/**
 * @brief mixin class for swarm communication
 *
 * @tparam T
 */
template <typename T>
class Communication {
 public:
  struct Params {
    const double normalized_communication_range = 0.2;
    const double delay = 0;  // step
  };

  Communication(Params params) : params_{params} {}
  virtual ~Communication() = default;

  auto derived() { return static_cast<T*>(this); }

  // KD-tree related members
  struct PointCloud {
    std::vector<std::array<qreal, 2>> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class
    inline qreal kdtree_get_pt(const size_t idx, const size_t dim) const {
      return pts[idx][dim];
    }

    // Optional bounding-box computation
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
      return false;
    }
  };

  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<qreal, PointCloud>,
      PointCloud,
      2>;
  std::shared_ptr<KDTree> kd_tree_;

  void buildKDTree() {
    cloud_.pts.clear();
    for (auto const& robot : derived()->robots()) {
      cloud_.pts.push_back({robot->x(), robot->y()});
    }

    if (!cloud_.pts.empty()) {
      kd_tree_ = std::make_shared<KDTree>(
          2, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
      kd_tree_->buildIndex();
    } else {
      kd_tree_.reset();
    }
  }

  std::vector<Robot*> robotsInRange(const Robot* robot) {
    return robotsInRange(
        {robot->x(), robot->y()},
        params_.normalized_communication_range * derived()->length());
  }

  void updateComm() {
    buildKDTree();
    comm_map_.clear();
    for (auto const& robot : derived()->robots()) {
      comm_map_[robot] = robotsInRange(robot);
    }
  }

  auto const& comm_map() const { return comm_map_; }

  const Params params_;

 private:
  std::vector<Robot*> robotsInRange(const QPointF& point, qreal range) {
    std::vector<Robot*> result;
    if (!kd_tree_) return result;

    std::vector<nanoflann::ResultItem<typename KDTree::IndexType, qreal>>
        ret_matches;
    nanoflann::SearchParameters params;
    const qreal query_pt[2] = {point.x(), point.y()};
    const size_t nMatches = kd_tree_->radiusSearch(
        &query_pt[0], range * range, ret_matches, params);

    for (size_t i = 0; i < nMatches; ++i) {
      result.push_back(derived()->robots()[ret_matches[i].first]);
    }

    return result;
  }

  PointCloud cloud_;
  using CommTable = std::map<Robot*, std::vector<Robot*>>;
  CommTable comm_map_;
};