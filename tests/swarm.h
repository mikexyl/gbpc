#pragma once

#include <gtsam/inference/Symbol.h>

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QWidget>
#include <vector>

#include "playground.h"
#include "robot.h"

class Swarm : public QGraphicsView {
  Q_OBJECT

 public:
  Swarm(QWidget* parent = nullptr) : QGraphicsView(parent) {
    scene = new QGraphicsScene(this);
    setScene(scene);
    setRenderHint(QPainter::Antialiasing);
    setSceneRect(-100, -100, 200, 200);
  }

  void addRobot(qreal x, qreal y, qreal radius) {
    Robot* robot = new Robot(gtsam::Symbol('p', num_robot_++), x, y, radius);
    scene->addItem(robot);
    robots.push_back(robot);
  }

  void removeRobot() {
    if (!robots.empty()) {
      Robot* robot = robots.back();
      scene->removeItem(robot);
      delete robot;
      robots.pop_back();
    }
  }

 private:
  QGraphicsScene* scene;
  std::vector<Robot*> robots;

  int num_robot_ = 0;
};
