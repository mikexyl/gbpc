#pragma once

#include <fmt/format.h>

#include <QGraphicsRectItem>
#include <random>

#include "robot.h"

class PlayGround : public QObject, public QGraphicsRectItem {
  Q_OBJECT

 public:
  PlayGround(qreal x,
             qreal y,
             qreal w,
             qreal h,
             QGraphicsItem* parent = nullptr)
      : QGraphicsRectItem(x, y, w, h, parent) {}
  virtual ~PlayGround() = default;

  Robot* spawn() {
    qreal radius = this->rect().width() / 300;
    // set random seed
    std::mt19937 gen(Robot::GetNumRobots());
    std::uniform_real_distribution<> dis_width(0, this->rect().width()),
        dis_height(0, this->rect().height());

    return new Robot(dis_width(gen), dis_height(gen), radius);
  }

 public slots:
  void setNewTarget(Robot* robot) {
    float min_size = std::min(this->rect().width(), this->rect().height());
    float max_size = std::max(this->rect().width(), this->rect().height());
    const float kMinMoveDistance = min_size * 0.1,
                kMaxMoveDistance = max_size * 0.2;
    static constexpr int kNStep = 100;

    // set random seed
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis_dist(kMinMoveDistance,
                                              kMaxMoveDistance);
    float r = dis_dist(gen);
    std::vector<QPointF> feasible_targets;
    for (float theta = 0; theta < 2 * M_PI; theta += M_PI / 20) {
      QPointF new_pos = robot->pos() + QPointF(r * cos(theta), r * sin(theta));
      if (this->contains(new_pos)) {
        feasible_targets.push_back(new_pos);
      }
    }

    // randomly choose a feasible target
    std::uniform_int_distribution<> dis_theta(0, feasible_targets.size() - 1);
    QPointF target = feasible_targets[dis_theta(gen)];
    robot->setTarget(target, kNStep);
  }
};