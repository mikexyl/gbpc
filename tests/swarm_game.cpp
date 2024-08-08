#include <gtsam/inference/Symbol.h>

#include <QApplication>
#include <QGraphicsLineItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QMainWindow>
#include <QTimer>
#include <random>

#include "playground.h"
#include "robot.h"

int Robot::NumRobots = 0;

class Swarm : public QGraphicsView {
  Q_OBJECT

 public:
  using Params = PlayGround::Params;

  Swarm(PlayGround::Params comm_params,
        int num_robots,
        QWidget* parent = nullptr)
      : QGraphicsView(parent) {
    scene = new QGraphicsScene(this);
    setScene(scene);
    setRenderHint(QPainter::Antialiasing);
    qreal width = 1600, height = 900;
    setSceneRect(0, 0, width, height);

    // draw a bounding box of the scene
    playground_ = new PlayGround(0, 0, width, height, comm_params);
    scene->addItem(playground_);

    addRobot(num_robots);

    // set a timer for updating communication
    QTimer* timer = new QTimer(this);
    connect(timer,
            &QTimer::timeout,
            this->playground_,
            &PlayGround::updateCommTimerEvent);
    connect(
        timer, &QTimer::timeout, this->playground_, &PlayGround::moveRobots);
    static int const kWindoFreq = 30;
    int const kUpdateCommInterval = 1000 / kWindoFreq;
    timer->start(kUpdateCommInterval);
  }

  void addRobot() { playground_->spawn(); }

  void addRobot(int num_robots) {
    for (int i = 0; i < num_robots; i++) {
      addRobot();
    }
  }

  void removeRobot() { playground_->removeRobot(); }

  auto const& robots() const { return playground_->robots(); }

 private:
  QGraphicsScene* scene;
  PlayGround* playground_;
};

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
    swarm = new Swarm(Swarm::Params{}, 30, this);
    setCentralWidget(swarm);
    setWindowTitle("Swarm Simulation");
    showMaximized();
  }

  void addRobot() { swarm->addRobot(); }

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