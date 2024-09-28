#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include "gbpc/factor.h"
#include "gbpc/gaussian.h"
#include "gbpc/graph.h"
#include "gbpc/variable.h"
#include "gbpc/visualization/visualization.h"

int main(int argc, char** argv) {
  gbpc::Graph graph;

  int num_nodes = 6;
  std::vector<gbpc::Belief<Point2>> init;
  for (int i = 0; i < num_nodes; i++) {
    init.emplace_back(gbpc::Belief<Point2>(i,
                                           Point2(i * 100 + 100, i * 100 + 100),
                                           Eigen::Matrix2d::Identity() * 100,
                                           0));
  }

  // Add variables
  std::vector<gbpc::Variable<Point2>::shared_ptr> variables;
  for (int i = 0; i < num_nodes; i++) {
    variables.push_back(std::make_shared<gbpc::Variable<Point2>>(init[i]));
  }

  for (int i = 0; i < num_nodes - 1; i++) {
    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();
    cov << 20, 0, 0, 20;
    cov *= cov;
    gbpc::Belief<Point2> measured(i + 10, Point2(50, 50), cov, 1);
    auto factor = std::make_shared<gbpc::BetweenFactor<Point2>>(measured);
    factor->addAdjVar({variables[i], variables[i + 1]});
    graph.add(factor);
  }

  auto prior_factor =
      std::make_shared<gbpc::PriorFactor<Point2>>(gbpc::Belief<Point2>(
          20, Point2(100, 100), Eigen::Matrix2d::Identity() * 100, 100));
  prior_factor->addAdjVar(variables[0]);
  graph.add(prior_factor);

  std::vector<gbpc::Gaussian> gtsam_results;

  sf::RenderWindow window;
  tgui::Gui gui;

  gbpc::visualization::createWindow(&window, &gui, [&graph, &gtsam_results]() {
    gtsam_results = graph.solveByGtsam();
  });
  sf::Clock clock;
  sf::Time press_time = sf::seconds(0.5f);  // Set cooldown time to 0.5 seconds
  sf::Time last_press_time = sf::Time::Zero;

  std::vector<sf::Color> colors;
  for (int i = 0; i < 1000; i++) {
    colors.emplace_back(gbpc::visualization::getColorFromValue(i, 0, 20));
  }

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) window.close();
      gui.handleEvent(event);  // Pass the event to the GUI
    }
    window.clear(sf::Color::White);

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space) &&
        clock.getElapsedTime() - last_press_time > press_time) {
      last_press_time = clock.getElapsedTime();
      graph.optimize();
    }

    // visualize the graph
    for (auto const& [_, var] : graph.vars()) {
      gbpc::visualization::plotEllipse(
          &window,
          gbpc::visualization::setAlpha(colors[var->key()], 0.3),
          *var);
    }

    for (auto const& factor : graph.factors()) {
      if (factor->adj_vars().size() == 2) {
        gbpc::visualization::plotConnectBeliefs(
            &window,
            gbpc::visualization::setAlpha(sf::Color::Black, 0.3),
            *(factor->adj_vars()[0]),
            *(factor->adj_vars()[1]));
      }
    }

    for (auto const& gaussian : gtsam_results) {
      gbpc::visualization::plotEllipse(
          &window,
          gbpc::visualization::setAlpha(sf::Color::Yellow, 0.3),
          gaussian);
    }

    gui.draw();  // Draw all widgets
    window.display();

    sf::sleep(sf::milliseconds(10));
  }

  std::cout << graph.print() << std::endl;
}
