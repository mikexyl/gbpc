#pragma once

#include <fmt/format.h>

#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <random>

#include "communication.h"
#include "robot.h"

class PlayGround : public QObject,
                   public QGraphicsRectItem,
                   public Communication<PlayGround> {
  Q_OBJECT

 public:
  struct Params : public Communication<PlayGround>::Params {
    Params() : Communication<PlayGround>::Params{} {}
    float min_move_distance_normalized = 0.05;
    float max_move_distance_normalized = 0.051;
    float max_delta_theta = M_PI / 3;
    int n_step = 30;
  };

  PlayGround(qreal x,
             qreal y,
             qreal w,
             qreal h,
             Params params = {},
             QGraphicsItem* parent = nullptr)
      : QGraphicsRectItem(x, y, w, h, parent),
        Communication<PlayGround>{params} {}
  virtual ~PlayGround() = default;

  float length() const { return std::max(rect().width(), rect().height()); }

  void spawn() {
    qreal radius = this->rect().width() / 300;
    // set random seed
    std::mt19937 gen(Robot::GetNumRobots());
    std::uniform_real_distribution<> dis_width(0, this->rect().width()),
        dis_height(0, this->rect().height()), dis_theta(0, 2 * M_PI);

    auto robot =
        new Robot(dis_width(gen), dis_height(gen), dis_theta(gen), radius);
    connect(robot, &Robot::finishedPath, this, &PlayGround::setNewTarget);
    connect(robot,
            &Robot::positionChanged,
            this,
            &PlayGround::robotPositionChanged);
    scene()->addItem(robot);

    QGraphicsPathItem* path_item = new QGraphicsPathItem();
    QColor robot_color_transparent(robot->color());
    robot_color_transparent.setAlphaF(0.3);
    path_item->setPen(QPen(robot_color_transparent, 2));
    path_item->setPos(robot->pos());
    scene()->addItem(path_item);
    path_item_[robot] = path_item;
    start_position_[robot] = robot->pos();
    robots_.push_back(robot);
  }

 public:
  const Params params_;

 public slots:
  void setNewTarget(Robot* robot, const Path& finished_path) {
    float min_size = std::min(this->rect().width(), this->rect().height());
    float max_size = std::max(this->rect().width(), this->rect().height());

    // set random seed
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis_dist(
        params_.min_move_distance_normalized * min_size,
        params_.max_move_distance_normalized * max_size),
        dist_dtheta(-params_.max_delta_theta, params_.max_delta_theta);
    float r = dis_dist(gen), dtheta = dist_dtheta(gen);
    float new_theta = robot->theta() + dtheta;

    QPointF new_pos =
        robot->pos() + QPointF(r * cos(new_theta), r * sin(new_theta));
    QPointF target;
    if (this->contains(new_pos)) {
      target = new_pos;
    } else {
      std::vector<QPointF> feasible_targets;
      std::vector<float> feasible_thetas;
      for (float theta = 0; theta < 2 * M_PI; theta += M_PI / 20) {
        QPointF new_pos =
            robot->pos() + QPointF(r * cos(theta), r * sin(theta));
        if (this->contains(new_pos)) {
          feasible_targets.push_back(new_pos);
          feasible_thetas.push_back(theta);
        }
      }
      // randomly choose a feasible target
      std::uniform_int_distribution<> dis_theta(0, feasible_targets.size() - 1);
      int rdn_idx = dis_theta(gen);
      target = feasible_targets[rdn_idx];
      new_theta = feasible_thetas[rdn_idx];
    }

    robot->setTarget(target, params_.n_step);
    robot->setTheta(new_theta);

    auto comm_map_it = comm_map().find(robot);
    if (comm_map_it == comm_map().end()) {
      return;
    }

    for (auto const& neighbor : comm_map_it->second) {
      if (not detect(robot, neighbor)) {
        continue;
      }
      QPointF p1 = robot->pos(), p2 = neighbor->pos();
      QLineF line(p1, p2);
      // draw the line
      scene()->addLine(
          line, QPen(Qt::black, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    }
  }

  void robotPositionChanged(Robot* robot) {
    auto path_item = path_item_[robot];
    QPainterPath path = path_item->path();
    path.lineTo(robot->pos() - start_position_[robot]);
    path_item->setPath(path);
  }

  const std::vector<Robot*>& robots() const { return robots_; }

  void removeRobot() {
    if (!robots_.empty()) {
      Robot* robot = robots_.back();
      scene()->removeItem(robot);
      delete robot;
      robots_.pop_back();
    }
  }

  void moveRobots() {
    for (auto robot : robots_) {
      robot->move();
    }
  }

  void updateCommTimerEvent() {
    updateComm();
    // remove old lines
    for (auto line : lines_) {
      scene()->removeItem(line);
      delete line;
    }

    lines_.clear();

    // add new lines
    for (auto const& [robot, neighbors] : comm_map()) {
      for (auto neighbor : neighbors) {
        QGraphicsLineItem* line = scene()->addLine(
            robot->x(),
            robot->y(),
            neighbor->x(),
            neighbor->y(),
            QPen(Qt::red, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        lines_.push_back(line);
      }
    }
  }

 private:
  std::map<Robot*, QGraphicsPathItem*> path_item_;
  std::map<Robot*, QPointF> start_position_;
  std::vector<Robot*> robots_;

  std::vector<QGraphicsLineItem*> lines_;
};