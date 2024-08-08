#pragma once

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>

#include <QGraphicsItem>
#include <QPainter>
#include <random>
#include <string>

struct Path {
  QPointF start;
  QPointF end;
  int total_steps{-1};
  int step{0};
};

class Robot : public QObject, public QGraphicsItem {
  Q_OBJECT
  Q_INTERFACES(QGraphicsItem)

 public:
  Robot(gtsam::Key id, qreal x, qreal y, qreal radius)
      : id_(id), radius_(radius) {
    setPos(x, y);

    std::mt19937 gen(id_);
    std::uniform_real_distribution<> dis_color(0, 255);
    color_ = QColor(dis_color(gen), dis_color(gen), dis_color(gen));
  }

  Robot(qreal x, qreal y, qreal radius)
      : Robot(gtsam::Symbol('p', NumRobots++), x, y, radius) {}

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
      emit finishedPath(this);
    }
  }

  void setTarget(QPointF target, int steps) { path_ = {pos(), target, steps}; }

  void setTarget(float theta, float r, int steps) {
    path_ = {pos(),
             pos() + QPointF(r * std::cos(theta), r * std::sin(theta)),
             steps};
  }

  auto color() const { return color_; }

  const gtsam::Key id_;

 signals:
  void finishedPath(Robot*);
  void positionChanged(Robot*);

 private:
  qreal radius_;
  Path path_;
  QColor color_;

 public:
  static int GetNumRobots() { return NumRobots; }

 private:
  static int NumRobots;
};
