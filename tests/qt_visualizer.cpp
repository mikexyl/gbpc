#include <tbb/concurrent_set.h>

#include <Eigen/Dense>  // Include Eigen for matrix operations
#include <QApplication>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QListWidget>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QTextBrowser>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <future>

#include "gbpc/gbpc.h"

class Ellipse {
 public:
  Ellipse(const Eigen::Vector2d& mean,
          const Eigen::Matrix2d& covariance,
          const Eigen::Vector3d& color = Eigen::Vector3d(0, 0, 0),
          bool draw_ellipse = true)
      : mean(mean),
        covariance(covariance),
        draw_ellipse(draw_ellipse),
        color(QColor(color[0], color[1], color[2], 50)) {
    calculateParameters();
  }

  void draw(QPainter& painter) const {
    painter.save();
    painter.translate(mean[0], mean[1]);

    if (draw_ellipse) {
      painter.setPen(Qt::red);
      painter.setBrush(color);  // Transparent red fill color
      painter.rotate(angle);
      painter.drawEllipse(QPointF(0, 0), axisLength1, axisLength2);
    }

    // draw mean dot
    painter.setPen(Qt::black);
    auto dot_color = this->color;
    dot_color.setAlpha(255);
    painter.setBrush(dot_color);
    painter.drawEllipse(QPointF(0, 0), 2, 2);

    painter.restore();
  }

  Eigen::Vector2d getMean() const { return mean; }

 private:
  Eigen::Vector2d mean;
  Eigen::Matrix2d covariance;
  double axisLength1;
  double axisLength2;
  double angle;
  bool draw_ellipse;
  QColor color;
  static QColor HighlightColor;

  void calculateParameters() {
    // Compute eigenvalues and eigenvectors of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
    Eigen::Vector2d eigenvalues = solver.eigenvalues();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors();

    // The lengths of the semi-axes are proportional to the square roots of the
    // eigenvalues
    axisLength1 =
        std::sqrt(eigenvalues[1]) * 2;  // Scale factor for visualization
    axisLength2 =
        std::sqrt(eigenvalues[0]) * 2;  // Scale factor for visualization

    // The angle of rotation of the ellipse
    angle = std::atan2(eigenvectors(1, 1), eigenvectors(0, 1)) * 180 / M_PI;
  }

 public:
  bool contains(const QPointF& point) const {
    // Inverse transform the point to the ellipse's coordinate system
    QTransform transform;
    transform.translate(mean[0], mean[1]);
    transform.rotate(angle);
    QPointF localPoint = transform.inverted().map(point);

    // Check if the point is within the ellipse using the ellipse equation
    double x = localPoint.x();
    double y = localPoint.y();
    return (x * x) / (axisLength1 * axisLength1) +
               (y * y) / (axisLength2 * axisLength2) <=
           1.0;
  }
};

class AnimatedWidget : public QWidget {
  Q_OBJECT

 public:
  static constexpr float dt = 0.03;
  static constexpr float duration = 0.2;

  AnimatedWidget(QWidget* parent = nullptr) : QWidget(parent) {
    startAnimation();
  }

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
    connect(timer, &QTimer::timeout, this, &AnimatedWidget::updatePosition);
    timer->start(dt * 1000);
  }

  void updatePosition() {
    update();  // Trigger a repaint
  }
};

class MainWindow : public QWidget {
  Q_OBJECT

 public:
  MainWindow(QWidget* parent = nullptr) : QWidget(parent) {
    mainLayout = new QHBoxLayout(this);

    // Sidebar with buttons
    sidebarLayout = new QVBoxLayout;

    sidebarLayout->addStretch();

    QGroupBox* sidebar = new QGroupBox("Controls");
    sidebar->setLayout(sidebarLayout);
    sidebar->setFixedWidth(150);  // Set fixed width for the sidebar

    // Animated widget
    animatedWidget = new AnimatedWidget;

    // Add sidebar and animated widget to the main layout
    mainLayout->addWidget(sidebar, 1);  // Sidebar with stretch factor 1
    mainLayout->addWidget(animatedWidget,
                          4);  // Animated widget with larger stretch factor

    // Set up the list widget
    listWidget = new QListWidget(this);

    // enable multi selection
    listWidget->setSelectionMode(QAbstractItemView::MultiSelection);

    connect(listWidget,
            &QListWidget::itemSelectionChanged,
            this,
            &MainWindow::handleMultiItemsSelected);

    // // Connect the itemClicked signal to a slot
    // connect(listWidget,
    //         &QListWidget::itemClicked,
    //         this,
    //         &MainWindow::handleItemClicked);

    QGroupBox* right_sidebar = new QGroupBox("");
    QVBoxLayout* right_sidebar_layout = new QVBoxLayout;
    right_sidebar_layout->addWidget(listWidget);
    right_sidebar->setLayout(right_sidebar_layout);

    mainLayout->addWidget(right_sidebar, 1);
  }

  void addButtonActions(std::string name,
                        std::function<void(QPushButton*)> action,
                        std::string text = "") {
    if (text.empty()) text = name;
    QPushButton* button = new QPushButton(QString::fromStdString(text));
    buttons[name] = button;
    sidebarLayout->addWidget(button);
    connect(
        buttons[name], &QPushButton::clicked, this, std::bind(action, button));
  }

  auto& addAnimation(const Eigen::Vector2d& start, const Eigen::Vector2d& end) {
    return animatedWidget->addAnimation(start, end);
  }

  auto& getListWidget() { return listWidget; }

 public slots:
  void addNewEllipse() {
    // Example mean and covariance
    Eigen::Vector2d mean(200, 200);
    Eigen::Matrix2d covariance;
    covariance << 400, 100, 100, 200;
    animatedWidget->addEllipse(mean, covariance);
  }

  void handleItemClicked(QListWidgetItem* item) {
    // Handle the item being clicked
    auto data = item->data(Qt::UserRole).value<gbpc::GBPClique::shared_ptr>();
    if (data) {
      animatedWidget->highlightCliques({data});
    }
  }

  void handleMultiItemsSelected() {
    // Handle the item being clicked
    std::set<gbpc::GBPClique::shared_ptr> cliques;
    for (auto item : listWidget->selectedItems()) {
      auto data = item->data(Qt::UserRole).value<gbpc::GBPClique::shared_ptr>();
      if (data) {
        cliques.insert(data);
      }
    }

    animatedWidget->highlightCliques(cliques);
  }

 public:
  void addEllipse(Eigen::Vector2d mean, Eigen::Matrix2d covariance) {
    animatedWidget->addEllipse(mean, covariance);
  }

  void addGraph(const gbpc::Graph::shared_ptr& graph,
                Eigen::Vector3d color,
                std::string name = "") {
    animatedWidget->addGraph(graph, color, name);
  }

 private:
  QHBoxLayout* mainLayout;
  QVBoxLayout* sidebarLayout;
  AnimatedWidget* animatedWidget;
  QListWidget* listWidget;

  std::map<std::string, QPushButton*> buttons;
};

inline Eigen::Vector2d asPoint(const gbpc::Node::shared_ptr& node) {
  Eigen::Vector2d point;
  if (auto sender_as_factor = std::dynamic_pointer_cast<gbpc::Factor>(node)) {
    if (auto between_factor =
            std::dynamic_pointer_cast<gbpc::BetweenFactor<Point2>>(
                sender_as_factor)) {
      point = (between_factor->var1()->mu() + between_factor->var2()->mu()) / 2;
    } else if (auto prior_factor =
                   std::dynamic_pointer_cast<gbpc::PriorFactor<Point2>>(
                       sender_as_factor)) {
      point = prior_factor->var()->mu();
    }
  } else {
    point = node->mu();
  }

  return point;
}

QColor Ellipse::HighlightColor = QColor(255, 0, 0, 100);

int main(int argc, char* argv[]) {
  gbpc::Graph::shared_ptr graph(new gbpc::Graph);

  // first arg is the number of nodes
  int num_nodes = 6;
  if (argc > 1) {
    num_nodes = std::stoi(argv[1]);
  }

  std::vector<gbpc::Belief<Point2>> init;
  std::vector<Eigen::Vector2d> samples;
  std::vector<Eigen::Vector2d> noised_samples;
  static constexpr int kRadius = 200;
  static constexpr float kNoise = 20;
  float kCov = kNoise * kNoise;
  for (int i = 0; i < num_nodes; i++) {
    double angle = 2 * M_PI * i / num_nodes;
    Eigen::Vector2d sample(kRadius * std::cos(angle) + kRadius + 200,
                           kRadius * std::sin(angle) + kRadius + 200);
    samples.push_back(sample);
    noised_samples.push_back(sample + Eigen::Vector2d::Random() * kNoise);
  }

  for (int i = 0; i < num_nodes; i++) {
    init.emplace_back(gbpc::Belief<Point2>(
        i, samples[i], Eigen::Matrix2d::Identity() * kCov * 3, 0));
  }

  // Add variables
  std::vector<gbpc::Variable<Point2>::shared_ptr> variables;
  for (int i = 0; i < num_nodes; i++) {
    variables.push_back(std::make_shared<gbpc::Variable<Point2>>(init[i]));
  }

  std::vector<gbpc::Factor::shared_ptr> loop_factors;
  for (int i = 0; i < num_nodes - 1; i++) {
    int i_next = i + 1;
    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity() * kCov;
    auto noise_measured = Point2(noised_samples[i_next] - noised_samples[i]);
    gbpc::Belief<Point2> measured(i + 10, noise_measured, cov, 1);
    auto factor = std::make_shared<gbpc::BetweenFactor<Point2>>(measured);
    factor->addAdjVar({variables[i], variables[i_next]});
    graph->add(factor);
  }

  auto prior_factor = std::make_shared<gbpc::PriorFactor<Point2>>(
      gbpc::Belief<Point2>(20, samples[0], Eigen::Matrix2d::Identity(), 100));
  prior_factor->addAdjVar(variables[0]);
  graph->add(prior_factor);

  QApplication app(argc, argv);

  MainWindow window;

  bool show_animation = true;

  gbpc::GBPUpdateParams params;
  params.root = prior_factor;
  params.post_pass = [&](const gbpc::Node::shared_ptr& sender,
                         const gbpc::Node::shared_ptr& receiver) {
    if (not show_animation) return;
    Eigen::Vector2d start = asPoint(sender);
    Eigen::Vector2d end = asPoint(receiver);
    window.addAnimation(start, end).wait(false);
  };

  std::future<void> gbp_opt_future;

  window.addGraph(graph, Eigen::Vector3d(0, 0, 255));
  window.addButtonActions("GBP Optimize", [&](QPushButton* button) {
    if ((not gbp_opt_future.valid()) or
        gbp_opt_future.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
      gbp_opt_future =
          std::async(std::launch::async, [&]() { graph->optimize(params); });
    } else {
      std::cout << "GBP Optimize is already running" << std::endl;
    }
  });
  window.addButtonActions(
      "LM Optimize", [&graph, &window](QPushButton* button) {
        auto lm_graph = graph->solveByGtsam();
        window.addGraph(lm_graph, Eigen::Vector3d(0, 255, 0), "LM");
      });
  window.addButtonActions("Reset", [&variables, &init](QPushButton* button) {
    for (int i = 0; i < variables.size(); i++) {
      variables[i]->replace(init[i]);
    }
  });

  window.addButtonActions("Bayes Tree", [&graph, &window](QPushButton* button) {
    graph->buildBayesTree();
    auto list_widget = window.getListWidget();
    list_widget->clear();
    for (const auto& clique : graph->cliques()) {
      auto item = new QListWidgetItem(list_widget);
      item->setText(QString::fromStdString(clique->print()));
      QVariant data;
      data.setValue(clique);
      item->setData(Qt::UserRole, data);
      list_widget->addItem(item);
    }
  });

  window.addButtonActions(
      "Optimize Roots", [&graph, &window](QPushButton* button) {
        graph->optimizeRoots();

        // message box
        QString message = "Optimize Roots Not Implemented";
        QMessageBox::information(&window, "Optimize Roots", message);
      });

  window.addButtonActions(
      "Optimize Selected", [&graph, &window](QPushButton* button) {
        auto list_widget = window.getListWidget();
        std::set<gbpc::GBPClique::shared_ptr> cliques;
        for (auto item : list_widget->selectedItems()) {
          auto data =
              item->data(Qt::UserRole).value<gbpc::GBPClique::shared_ptr>();
          if (data) {
            cliques.insert(data);
          }
        }

        // message box
        QString message = "Optimize Selected Cliques Not Implemented";
        QMessageBox::information(&window, "Optimize Selected", message);
      });

  window.addButtonActions("Toggle Loop",
                          [&graph, &loop_factors](QPushButton* button) {
                            button->setCheckable(true);
                            if (graph->hasAny(loop_factors)) {
                              graph->remove(loop_factors);
                              button->setChecked(false);
                            } else {
                              graph->add(loop_factors);
                              button->setChecked(true);
                            }
                          });

  window.addButtonActions(
      "Add Loop",
      [&graph, &variables, &loop_factors, kCov](QPushButton* button) {
        int i = random() % variables.size();
        int i_next = i;
        do {
          i_next = random() % variables.size();
        } while (i_next == i or std::abs(i_next - i) == 1);

        Eigen::Matrix2d cov = Eigen::Matrix2d::Identity() * kCov;
        auto noise_measured =
            Point2(variables[i_next]->mu() - variables[i]->mu());
        // more noise
        noise_measured += Point2::Random() * 10;
        gbpc::Belief<Point2> measured(i + 10, noise_measured, cov, 1);
        auto factor = std::make_shared<gbpc::BetweenFactor<Point2>>(measured);
        factor->addAdjVar({variables[i], variables[i_next]});
        loop_factors.push_back(factor);
        graph->add(factor);
      });

  window.addButtonActions(
      "Toggle Animation",
      [&show_animation](QPushButton* button) {
        button->setCheckable(true);
        show_animation = not show_animation;
        button->setChecked(show_animation);
      },
      "Toggle Animation");

  // full screen
  window.showMaximized();
  window.show();

  return app.exec();
}

#include "qt_visualizer.moc"
