#pragma once

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

#include <QGraphicsItem>
#include <QPainter>
#include <random>

struct Path {
  QPointF start;
  QPointF end;
  int total_steps{-1};
  int step{0};

  gtsam::BetweenFactor<gtsam::Pose2>::shared_ptr toFactor() {
    return boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
        TrajectoryKey++,
        TrajectoryKey++,
        gtsam::Pose2(start.x(), start.y(), 0),
        gtsam::noiseModel::Unit::Create(3));
  }

  static std::atomic<size_t> TrajectoryKey;
};

class Robot : public QObject, public QGraphicsItem {
  Q_OBJECT
  Q_INTERFACES(QGraphicsItem)

 public:
  Robot(gtsam::Key id, qreal x, qreal y, qreal theta, qreal radius)
      : id_(id), theta_(theta), radius_(radius) {
    new_inter_factors_ = boost::make_shared<gtsam::NonlinearFactorGraph>();
    new_intra_factors_ = boost::make_shared<gtsam::NonlinearFactorGraph>();

    setPos(x, y);

    std::mt19937 gen(id_);
    std::uniform_real_distribution<> dis_color(0, 255);
    color_ = QColor(dis_color(gen), dis_color(gen), dis_color(gen));
  }

  Robot(qreal x, qreal y, qreal theta, qreal radius)
      : Robot(gtsam::Symbol('p', NumRobots++), x, y, theta, radius) {}

  QRectF boundingRect() const override {
    // Define the bounding rectangle for the robot
    return QRectF(-radius_, -radius_, 2 * radius_, 2 * radius_);
  }

  void paint(QPainter* painter,
             const QStyleOptionGraphicsItem* option,
             QWidget* widget) override {
    Q_UNUSED(option);
    Q_UNUSED(widget);

    // Draw the robot as a circle
    painter->setBrush(color_);
    painter->drawEllipse(boundingRect());
    // draw the id as label
    painter->drawText(boundingRect(),
                      Qt::AlignCenter,
                      QString::fromStdString(gtsam::DefaultKeyFormatter(id_)));
    // draw a small dot to show theta
    QPointF dot =
        boundingRect().center() + QPointF(radius_ * 1.5 * std::cos(theta_),
                                          radius_ * 1.5 * std::sin(theta_));
    painter->drawEllipse(dot, radius_ / 1.5, radius_ / 1.5);
  }

  // Method to move the robot
  void move(qreal dx, qreal dy) {
    setPos(x() + dx, y() + dy);
    positionChanged(this);
  }

  /**
   * @brief move along the path
   *
   */
  void move() {
    if (path_.step < path_.total_steps) {
      qreal dx = (path_.end.x() - path_.start.x()) / path_.total_steps;
      qreal dy = (path_.end.y() - path_.start.y()) / path_.total_steps;
      move(dx, dy);
      path_.step++;
    } else {
      // new_intra_factors_->push_back(path_.toFactor());
      emit finishedPath(this, path_);
    }
  }

  void addInterRobotFactor(const gtsam::NonlinearFactor::shared_ptr& factor) {
    new_intra_factors_->push_back(factor);
  }

  void setTarget(QPointF target, int steps) { path_ = {pos(), target, steps}; }

  void setTarget(float theta, float r, int steps) {
    path_ = {pos(),
             pos() + QPointF(r * std::cos(theta), r * std::sin(theta)),
             steps};
  }

  auto color() const { return color_; }

  auto getNewFactors() {
    gtsam::NonlinearFactorGraph copy(*new_intra_factors_);
    new_intra_factors_.reset(new gtsam::NonlinearFactorGraph());
    return copy;
  }

  void setTheta(float theta) { theta_ = theta; }

  float theta() const { return theta_; }

  const gtsam::Key id_;

 signals:
  void finishedPath(Robot*, const Path& finished_path);
  void positionChanged(Robot*);

 private:
  float theta_;

  qreal radius_;
  Path path_;
  QColor color_;

  gtsam::NonlinearFactorGraph::shared_ptr new_intra_factors_;
  gtsam::NonlinearFactorGraph::shared_ptr new_inter_factors_;

 public:
  static int GetNumRobots() { return NumRobots; }

 private:
  static int NumRobots;
};
