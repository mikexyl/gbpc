#pragma once

#include <tbb/concurrent_set.h>

#include <Eigen/Eigen>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QTimer>
#include <QWidget>

#include "ellipse.h"
#include "gbpc/gbpc.h"

class GraphWidget : public QWidget {
  Q_OBJECT

 public:
  static constexpr float dt = 0.03;
  static constexpr float duration = 0.2;

  GraphWidget(QWidget* parent = nullptr);

  void addEllipse(const Eigen::Vector2d& mean,
                  const Eigen::Matrix2d& covariance,
                  const Eigen::Vector3d& color = Eigen::Vector3d(0, 0, 0),
                  bool draw_ellipse = true) {
    ellipses.emplace_back(mean, covariance, color, draw_ellipse);
    update();  // Trigger a repaint to show the new ellipse
  }

  void addGraph(const gbpc::Graph::shared_ptr& graph,
                Eigen::Vector3d color,
                std::string name = "") {
    if (name.size()) {
      if (named_graphs_.find(name) != named_graphs_.end()) {
        graphs_.erase(named_graphs_[name]);
        colors.erase(named_graphs_[name]);
      }
      named_graphs_[name] = graph;
    }

    graphs_.insert(graph);
    colors[graph] = color;

    update();
  }

  auto& addAnimation(const Eigen::Vector2d& start, const Eigen::Vector2d& end) {
    std::cout << "addAnimation from " << start.transpose() << " to "
              << end.transpose() << std::endl;
    auto task = std::make_shared<AnimationTask>(start, end, duration, dt);
    animation_tasks.insert(task);
    return task->finished;
  }

  QPainterPath cliquePath(const gbpc::GBPClique& clique) {
    QPainterPath path;

    if (clique.empty()) {
      return path;  // Return an empty path if there are no points
    }

    // Start the path at the first point
    path.moveTo(clique[0]->mu().x(), clique[0]->mu().y());

    for (int i = 1; i < clique.size(); ++i) {
      // Connect the nodes with a cubic Bezier curve for ultra-slick lines
      QPointF mid1((clique[i - 1]->mu().x() + clique[i]->mu().x()) / 2,
                   clique[i - 1]->mu().y());
      QPointF mid2(clique[i]->mu().x(),
                   (clique[i - 1]->mu().y() + clique[i]->mu().y()) / 2);
      path.cubicTo(
          mid1, mid2, QPointF(clique[i]->mu().x(), clique[i]->mu().y()));
    }

    // Optionally close the path if you want to connect the last node back to
    // the first
    path.lineTo(clique[0]->mu().x(), clique[0]->mu().y());

    return path;
  }

  void highlightCliques(std::set<gbpc::GBPClique::shared_ptr> cliques) {
    highlighted_cliques.clear();
    highlighted_cliques = cliques;
    update();
  }

 protected:
  void paintEvent(QPaintEvent* event) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw all ellipses
    for (const auto& ellipse : ellipses) {
      ellipse.draw(painter);
    }

    // draw graph
    for (const auto& graph : graphs_) {
      for (const auto& [_, var] : graph->vars()) {
        Ellipse ellipse(var->mu(), var->Sigma(), colors[graph], var->N() != 0);
        ellipse.draw(painter);
      }
    }

    // connect graph
    for (const auto& graph : graphs_) {
      for (const auto& factor : graph->factors()) {
        auto adj_vars = factor->adj_vars();
        for (int i = 0; i < adj_vars.size() - 1; i++) {
          connectPoints(painter, adj_vars[i]->mu(), adj_vars[i + 1]->mu());
        }
      }
    }

    for (auto it = animation_tasks.begin(); it != animation_tasks.end();) {
      auto& task = *it;
      if (task->finished) {
        it = animation_tasks.unsafe_erase(it);
        continue;
      }

      painter.save();
      auto current = task->step();
      painter.setPen(QPen(QColor(255, 0, 0), 2));
      painter.setBrush(QColor(255, 0, 0));
      painter.translate(current[0], current[1]);
      painter.drawEllipse(QPointF(0, 0), 5, 5);

      painter.restore();

      it++;
    }

    for (auto const& graph : graphs_) {
      for (auto const& clique : graph->cliques()) {
        painter.save();
        // set same seed
        srand(clique->front()->key());

        QColor random_color =
            QColor(rand() % 256, rand() % 256, rand() % 256, 10);
        if (highlighted_cliques.find(clique) != highlighted_cliques.end()) {
          random_color.setAlpha(40);
        }
        painter.setPen(QPen(random_color, 10));
        painter.setBrush(QColor(random_color));

        // Draw the lines connecting the nodes
        for (int i = 1; i < clique->size(); ++i) {
          painter.drawLine(clique->at(i - 1)->mu().x(),
                           clique->at(i - 1)->mu().y(),
                           clique->at(i)->mu().x(),
                           clique->at(i)->mu().y());
        }

        // Optionally close the path if you want to connect the last node back
        // to the first
        painter.drawLine(clique->back()->mu().x(),
                         clique->back()->mu().y(),
                         clique->front()->mu().x(),
                         clique->front()->mu().y());

        painter.restore();
      }
    }
  }

  void connectPoints(QPainter& painter,
                     const Eigen::Vector2d& point1,
                     const Eigen::Vector2d& point2) {
    painter.drawLine(point1[0], point1[1], point2[0], point2[1]);
  }

  void mousePressEvent(QMouseEvent* event) override {
    QPointF clickPos = event->pos();
    for (const auto& ellipse : ellipses) {
      if (ellipse.contains(clickPos)) {
        // Handle the ellipse being clicked (e.g., highlight it)
        qDebug() << "Ellipse clicked at" << clickPos;
        break;  // Remove this break if you want to detect multiple ellipses
                // under the click
      }
    }
  }

 private:
  QTimer* timer;
  std::vector<Ellipse> ellipses;
  std::set<gbpc::Graph::shared_ptr> graphs_;
  std::map<std::string, gbpc::Graph::shared_ptr> named_graphs_;
  std::map<gbpc::Graph::shared_ptr, Eigen::Vector3d> colors;
  std::set<gbpc::GBPClique::shared_ptr> highlighted_cliques;

  struct AnimationTask {
    AnimationTask(const Eigen::Vector2d& start,
                  const Eigen::Vector2d& end,
                  float duration,
                  float dt)
        : duration(duration), dt(dt), start(start), end(end) {}

    float duration{0.5};
    float dt{0.03};

    const Eigen::Vector2d start;
    const Eigen::Vector2d end;
    float t{0};

    Eigen::Vector2d step() {
      auto current = (end - start) * t / duration + start;

      if (t < duration) {
        t += dt;
      }
      if (t >= duration) {
        finished.store(true);
        finished.notify_all();
      }

      return current;
    }

    std::atomic<bool> finished{false};
  };
  tbb::concurrent_set<std::shared_ptr<AnimationTask>> animation_tasks;

  void startAnimation() {
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &GraphWidget::updatePosition);
    timer->start(dt * 1000);
  }

  void updatePosition() {
    update();  // Trigger a repaint
  }
};
