#include <gtsam/inference/Symbol.h>

#include <QApplication>
#include <QGraphicsLineItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QMainWindow>
#include <QTimer>
#include <random>

#include "nanoflann.hpp"
#include "robot.h"

/**
 * @brief mixin class for swarm communication
 *
 * @tparam T
 */
template <typename T>
class Communication {
 public:
  struct Params {
    const double communication_range = 300.0;
    const double delay = 0;  // step
  };

  Communication(Params params) : params_{params} {}
  virtual ~Communication() = default;

  auto swarm() { return static_cast<T*>(this); }

  // KD-tree related members
  struct PointCloud {
    std::vector<std::array<qreal, 2>> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class
    inline qreal kdtree_get_pt(const size_t idx, const size_t dim) const {
      return pts[idx][dim];
    }

    // Optional bounding-box computation
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
      return false;
    }
  };

  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<qreal, PointCloud>,
      PointCloud,
      2>;
  std::shared_ptr<KDTree> kd_tree_;

  void buildKDTree() {
    cloud_.pts.clear();
    for (auto const& robot : swarm()->robots()) {
      cloud_.pts.push_back({robot->x(), robot->y()});
    }

    if (!cloud_.pts.empty()) {
      kd_tree_ = std::make_shared<KDTree>(
          2, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
      kd_tree_->buildIndex();
    } else {
      kd_tree_.reset();
    }
  }

  std::vector<Robot*> robotsInRange(const Robot* robot) {
    return robotsInRange({robot->x(), robot->y()}, params_.communication_range);
  }

  void updateComm() {
    buildKDTree();
    comm_map_.clear();
    for (auto const& robot : swarm()->robots()) {
      comm_map_[robot] = robotsInRange(robot);
    }
  }

  auto const& comm_map() const { return comm_map_; }

  const Params params_;

 private:
  std::vector<Robot*> robotsInRange(const QPointF& point, qreal range) {
    std::vector<Robot*> result;
    if (!kd_tree_) return result;

    std::vector<nanoflann::ResultItem<typename KDTree::IndexType, qreal>>
        ret_matches;
    nanoflann::SearchParameters params;
    const qreal query_pt[2] = {point.x(), point.y()};
    const size_t nMatches = kd_tree_->radiusSearch(
        &query_pt[0], range * range, ret_matches, params);

    for (size_t i = 0; i < nMatches; ++i) {
      result.push_back(swarm()->robots()[ret_matches[i].first]);
    }

    return result;
  }

  PointCloud cloud_;
  using CommTable = std::map<Robot*, std::vector<Robot*>>;
  CommTable comm_map_;
};

class Swarm : public QGraphicsView, public Communication<Swarm> {
  Q_OBJECT

 public:
  Swarm(Communication<Swarm>::Params comm_params,
        int num_robots,
        QWidget* parent = nullptr)
      : QGraphicsView(parent), Communication<Swarm>{comm_params} {
    scene = new QGraphicsScene(this);
    setScene(scene);
    setRenderHint(QPainter::Antialiasing);
    qreal width = 1080, height = 720;
    setSceneRect(0, 0, width, height);

    // draw a bouding box of the scene
    scene->addRect(0, 0, width, height);

    std::cout << "width: " << width << ", height: " << height << std::endl;
    for (int i = 0; i < num_robots; i++) {
      // set random seed as i
      std::mt19937 gen(i);
      std::uniform_real_distribution<> dis_width(0, width),
          dis_height(0, height);

      addRobot(dis_width(gen), dis_height(gen), 10);
    }

    // set a timer for updating communication
    QTimer* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &Swarm::updateCommTimerEvent);
    static int const kUpdateCommInterval = 100;
    timer->start(kUpdateCommInterval);
  }

  void addRobot(qreal x, qreal y, qreal radius) {
    std::cout << "addRobot: " << num_robot_ << " " << x << " " << y << " "
              << radius << std::endl;
    Robot* robot = new Robot(gtsam::Symbol('p', num_robot_++), x, y, radius);
    scene->addItem(robot);
    robots_.push_back(robot);
  }

  void removeRobot() {
    if (!robots_.empty()) {
      Robot* robot = robots_.back();
      scene->removeItem(robot);
      delete robot;
      robots_.pop_back();
    }
  }

  auto const& robots() const { return robots_; }

  void updateCommTimerEvent() {
    updateComm();
    // remove old lines
    for (auto line : lines_) {
      scene->removeItem(line);
      delete line;
    }

    lines_.clear();

    // add new lines
    for (auto const& [robot, neighbors] : comm_map()) {
      for (auto neighbor : neighbors) {
        QGraphicsLineItem* line = scene->addLine(
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
  QGraphicsScene* scene;
  std::vector<Robot*> robots_;
  std::vector<QGraphicsLineItem*> lines_;

  int num_robot_ = 0;
};

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
    swarm = new Swarm(Swarm::Params{}, 20, this);
    setCentralWidget(swarm);
    setWindowTitle("Swarm Simulation");
    showMaximized();
  }

  void addRobot(qreal x, qreal y, qreal radius) {
    swarm->addRobot(x, y, radius);
  }

  void removeRobot() { swarm->removeRobot(); }

 private:
  Swarm* swarm;
};

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);

  MainWindow mainWindow;

  return app.exec();
}

#include "swarm_game.moc"