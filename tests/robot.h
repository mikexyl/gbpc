#pragma once

#include <gtsam/inference/Key.h>

#include <QGraphicsItem>
#include <QPainter>
#include <string>

class Robot : public QGraphicsItem {
 public:
  Robot(gtsam::Key id, qreal x, qreal y, qreal radius) : id_(id), radius_(radius) {
    setPos(x, y);
  }

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
    painter->setBrush(Qt::blue);
    painter->drawEllipse(boundingRect());
  }

  // Method to move the robot
  void move(qreal dx, qreal dy) { setPos(x() + dx, y() + dy); }

  const gtsam::Key id_;

 private:
  qreal radius_;
};
