#include <Eigen/Dense>  // Include Eigen for matrix operations
#include <QApplication>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include "gbpc/gbpc.h"

class Ellipse {
 public:
  Ellipse(const Eigen::Vector2d& mean,
          const Eigen::Matrix2d& covariance,
          const Eigen::Vector3d& color = Eigen::Vector3d(0, 0, 0))
      : mean(mean),
        covariance(covariance),
        color(QColor(color[0], color[1], color[2], 50)) {
    calculateParameters();
  }

  void draw(QPainter& painter) const {
    painter.save();
    painter.setPen(Qt::red);
    painter.setBrush(color);  // Transparent red fill color
    painter.translate(mean[0], mean[1]);
    painter.rotate(angle);
    painter.drawEllipse(QPointF(0, 0), axisLength1, axisLength2);
    painter.restore();
  }

  Eigen::Vector2d getMean() const { return mean; }

 private:
  Eigen::Vector2d mean;
  Eigen::Matrix2d covariance;
  double axisLength1;
  double axisLength2;
  double angle;
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
  AnimatedWidget(QWidget* parent = nullptr) : QWidget(parent), rectX(0) {
    startAnimation();
  }

  void addEllipse(const Eigen::Vector2d& mean,
                  const Eigen::Matrix2d& covariance) {
    ellipses.emplace_back(mean, covariance);
    update();  // Trigger a repaint to show the new ellipse
  }

  void addGraph(const gbpc::Graph::shared_ptr& graph, Eigen::Vector3d color) {
    graph_.emplace_back(graph);
    colors[graph] = color;
    update();
  }

 protected:
  void paintEvent(QPaintEvent* event) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw the moving rectangle
    QRect rect = QRect(rectX, 50, 100, 100);
    painter.setBrush(Qt::blue);
    painter.drawRect(rect);

    // Draw all ellipses
    for (const auto& ellipse : ellipses) {
      ellipse.draw(painter);
    }

    // draw graph
    for (const auto& graph : graph_) {
      for (const auto& [_, var] : graph->vars()) {
        Ellipse ellipse(var->mu(), var->Sigma(), colors[graph]);
        ellipse.draw(painter);
      }
    }

    // connect graph
    for (const auto& graph : graph_) {
      for (const auto& factor : graph->factors()) {
        auto adj_vars = factor->adj_vars();
        for (int i = 0; i < adj_vars.size() - 1; i++) {
          connectPoints(painter, adj_vars[i]->mu(), adj_vars[i + 1]->mu());
        }
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
  int rectX;
  QTimer* timer;
  std::vector<Ellipse> ellipses;
  std::vector<gbpc::Graph::shared_ptr> graph_;
  std::map<gbpc::Graph::shared_ptr, Eigen::Vector3d> colors;

  void startAnimation() {
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &AnimatedWidget::updatePosition);
    timer->start(30);  // Update every 30 ms
  }

  void updatePosition() {
    rectX += 5;  // Move 5 pixels to the right
    if (rectX > width()) {
      rectX = -100;  // Reset position when off screen
    }
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
  }

  void addButtonActions(std::string name, std::function<void()> action) {
    QPushButton* button = new QPushButton(QString::fromStdString(name));
    buttons[name] = button;
    sidebarLayout->addWidget(button);
    buttonActions[name] = action;
    connect(buttons[name], &QPushButton::clicked, this, action);
  }

 public slots:
  void addNewEllipse() {
    // Example mean and covariance
    Eigen::Vector2d mean(200, 200);
    Eigen::Matrix2d covariance;
    covariance << 400, 100, 100, 200;
    animatedWidget->addEllipse(mean, covariance);
  }

 public:
  void addEllipse(Eigen::Vector2d mean, Eigen::Matrix2d covariance) {
    animatedWidget->addEllipse(mean, covariance);
  }

  void addGraph(const gbpc::Graph::shared_ptr& graph, Eigen::Vector3d color) {
    animatedWidget->addGraph(graph, color);
  }

 private:
  QHBoxLayout* mainLayout;
  QVBoxLayout* sidebarLayout;
  AnimatedWidget* animatedWidget;

  std::map<std::string, QPushButton*> buttons;
  std::map<std::string, std::function<void()>> buttonActions;
};

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
        i, samples[i], Eigen::Matrix2d::Identity() * kCov * 3, 1));
  }

  // Add variables
  std::vector<gbpc::Variable<Point2>::shared_ptr> variables;
  for (int i = 0; i < num_nodes; i++) {
    variables.push_back(std::make_shared<gbpc::Variable<Point2>>(init[i]));
  }

  for (int i = 0; i < num_nodes; i++) {
    int i_next = (i + 1) % num_nodes;
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

  window.addGraph(graph, Eigen::Vector3d(0, 0, 255));
  window.addButtonActions("GBP Optimize", [&graph, &prior_factor]() {
    graph->optimize(prior_factor);
  });
  window.addButtonActions("LM Optimize", [&graph, &window]() {
    auto lm_graph = graph->solveByGtsam();
    window.addGraph(lm_graph, Eigen::Vector3d(0, 255, 0));
  });
  window.addButtonActions("Reset", [&variables, &init]() {
    for (int i = 0; i < variables.size(); i++) {
      variables[i]->replace(init[i]);
    }
  });

  // full screen
  window.showMaximized();
  window.show();

  return app.exec();
}

#include "qt_visualizer.moc"
