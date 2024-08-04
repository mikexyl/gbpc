#include <graphviz/gvc.h>
#include <tbb/concurrent_set.h>

#include <Eigen/Dense>  // Include Eigen for matrix operations
#include <QApplication>
#include <QGraphicsEllipseItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsView>
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

#include "ellipse.h"
#include "gbpc/gbpc.h"
#include "gbpc/graph.h"
#include "graph_widget.h"

class ClickableEllipseItem : public QGraphicsObject {
  Q_OBJECT

 public:
  ClickableEllipseItem(qreal x,
                       qreal y,
                       qreal width,
                       qreal height,
                       const gbpc::Key& key,
                       const QString& label,
                       QGraphicsItem* parent = nullptr)
      : QGraphicsObject(parent),
        key(key),
        label(label),
        ellipseRect(x, y, width, height) {
    setFlag(QGraphicsItem::ItemIsSelectable);
  }

  QRectF boundingRect() const override { return ellipseRect; }

  const gbpc::Key key;
  const QString label;

 protected:
  void mousePressEvent(QGraphicsSceneMouseEvent* event) override {
    if (contains(event->pos())) {
      qDebug() << "Ellipse clicked at:" << event->pos();
    }
    QGraphicsObject::mousePressEvent(event);
  }
  void paint(QPainter* painter,
             const QStyleOptionGraphicsItem* option,
             QWidget* widget) override {
    Q_UNUSED(option);
    Q_UNUSED(widget);

    painter->setBrush(Qt::lightGray);
    painter->drawEllipse(ellipseRect);
    painter->setPen(QPen(Qt::black, 1));
    painter->drawText(ellipseRect, Qt::AlignCenter, label);
  }

 private:
  QRectF ellipseRect;
};

class TreeWidget : public QWidget {
  Q_OBJECT

 public:
  TreeWidget(QWidget* parent = nullptr) : QWidget(parent) {
    scene = new QGraphicsScene(this);
    view = new QGraphicsView(scene, this);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(view);
    setLayout(layout);
  }

 public slots:
  void setTreeStr(const QString& graph_str) {
    this->graph_str_ = graph_str.toStdString();

    GVC_t* gvc = gvContext();
    Agraph_t* g = agmemread(graph_str.toStdString().c_str());
    gvLayout(gvc, g, "dot");
    char* result = nullptr;
    unsigned int length = 0;
    gvRenderData(gvc, g, "plain", &result, &length);
    setLayoutString(QString::fromStdString(result));

    graph_updated_ = true;
    update();
  }

 protected:
  void clear() {
    nodeLabels.clear();
    nodePositions.clear();
    edges.clear();
    update();
  }

  void paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    if (not graph_updated_) return;
    graph_updated_ = false;

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    auto step_x = width() / (numLevelX + 1);
    auto step_y = height() / (numLevelY + 1);

    nodes.clear();
    scene->clear();

    // draw edges
    for (auto edge : edges) {
      auto first = nodePositions[edge.first],
           second = nodePositions[edge.second];
      QPointF from = toWindowCoordinates(first) +
                     QPointF(step_x / 2., step_y / 2.),
              to = toWindowCoordinates(second) +
                   QPointF(step_x / 2., step_y / 2.);
      // draw arrow
      drawArrow(painter, from, to);
    }

    for (auto it = nodePositions.begin(); it != nodePositions.end(); ++it) {
      QPointF pos = toWindowCoordinates(it.value());
      QRectF ellipseRect(pos.x(), pos.y(), step_x, step_y * 0.6);
      // std string to size_t
      size_t key = std::stoul(it.key().toStdString());
      ClickableEllipseItem* node =
          new ClickableEllipseItem(ellipseRect.x(),
                                   ellipseRect.y(),
                                   ellipseRect.width(),
                                   ellipseRect.height(),
                                   key,
                                   nodeLabels[it.key()]);
      scene->addItem(node);
      std::cout << ellipseRect.x() << " " << ellipseRect.y() << " "
                << ellipseRect.width() << " " << ellipseRect.height()
                << std::endl;
    }
  }

  QPointF toWindowCoordinates(QPointF pos) {
    // Adjust positions for the widget's coordinate system
    pos.setX(pos.x() * this->width() * 0.5);
    pos.setY(pos.y() * this->height() * 0.5);
    return QPointF(pos.x(), pos.y());
  }

  void drawArrow(QPainter& painter, const QPointF& from, const QPointF& to) {
    Q_UNUSED(painter);
    scene->addLine(from.x(), from.y(), to.x(), to.y());

    // TODO: Draw arrowhead
  }

 private:
  std::string graph_str_;
  std::atomic<bool> graph_updated_{false};
  QMap<QString, QPointF> nodePositions;
  QVector<QPair<QString, QString>> edges;
  QMap<QString, QString> nodeLabels;
  QVector<ClickableEllipseItem*> nodes;
  int numLevelX = 0, numLevelY = 0;

  QGraphicsScene* scene;
  QGraphicsView* view;

  void setLayoutString(const QString& layout) {
    clear();
    parsePlainLayout(layout);
    update();  // Trigger a repaint
  }

  void parsePlainLayout(const QString& layout) {
    std::cout << layout.toStdString() << std::endl;
    nodePositions.clear();
    std::istringstream iss(layout.toStdString());
    std::string line;

    while (std::getline(iss, line)) {
      std::istringstream lineStream(line);
      std::string type;
      lineStream >> type;

      if (type == "node") {
        std::string name;
        double x, y, width, height;
        std::string label, style, shape, color, fillcolor;

        lineStream >> name >> x >> y >> width >> height;
        std::getline(lineStream, label, '\"');
        std::getline(lineStream,
                     label,
                     '\"');  // This is to skip to the actual label content
        lineStream >> style >> shape >> color >> fillcolor;

        nodePositions[QString::fromStdString(name)] = QPointF(x, y);
        nodeLabels[QString::fromStdString(name)] =
            QString::fromStdString(label);
      } else if (type == "edge") {
        std::string from, to;
        lineStream >> from >> to;
        edges.push_back(
            {QString::fromStdString(from), QString::fromStdString(to)});
      }
    }

    // normalize positions
    qreal min_x = std::numeric_limits<qreal>::max();
    qreal min_y = std::numeric_limits<qreal>::max();
    qreal max_x = std::numeric_limits<qreal>::min();
    qreal max_y = std::numeric_limits<qreal>::min();

    std::set<qreal> levelsX, levelsY;

    for (auto it = nodePositions.begin(); it != nodePositions.end(); ++it) {
      min_x = std::min(min_x, it.value().x());
      min_y = std::min(min_y, it.value().y());
      max_x = std::max(max_x, it.value().x());
      max_y = std::max(max_y, it.value().y());
      levelsX.insert(it.value().x());
      levelsY.insert(it.value().y());
    }

    qreal width = max_x - min_x, height = max_y - min_y;

    for (auto it = nodePositions.begin(); it != nodePositions.end(); ++it) {
      QPointF& pos = it.value();
      if (width != 0)
        pos.setX((pos.x() - min_x) / width);
      else
        pos.setX(0.5);
      if (height != 0)
        pos.setY(1 - (pos.y() - min_y) / height);
      else
        pos.setY(0.5);
    }

    numLevelX = levelsX.size();
    numLevelY = levelsY.size();
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
    animatedWidget = new GraphWidget;

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

    QGroupBox* right_sidebar = new QGroupBox("");
    QVBoxLayout* right_sidebar_layout = new QVBoxLayout;
    right_sidebar_layout->addWidget(listWidget);
    right_sidebar->setLayout(right_sidebar_layout);

    mainLayout->addWidget(right_sidebar, 1);

    // new window
    treeWidget = new TreeWidget(nullptr);
    treeWidget->show();

    connect(this, &MainWindow::setTreeStr, treeWidget, &TreeWidget::setTreeStr);
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

 signals:
  void setTreeStr(const QString& tree_str);

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
  GraphWidget* animatedWidget;
  QListWidget* listWidget;
  TreeWidget* treeWidget;

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
    auto tree_str = graph->buildBayesTree()->dot();
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

    // save str to file
    std::ofstream file("bayes_tree.dot");
    file << tree_str;
    file.close();
    window.setTreeStr(QString::fromStdString(tree_str));
  });

  window.addButtonActions(
      "Optimize Roots", [&graph, &window](QPushButton* button) {
        try {
          graph->optimizeRoots();
        } catch (const std::exception& e) {
          QMessageBox::information(&window, "Optimize Roots", e.what());
        }
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
