#ifndef GBPC_VISUALIZATION_VISUALIZATION_H_
#define GBPC_VISUALIZATION_VISUALIZATION_H_

#include <Eigen/Eigen>
#include <SFML/Graphics.hpp>
#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include <iostream>

#include "gbpc/gaussian.h"

namespace gbpc::visualization {

inline std::pair<std::vector<double>, std::vector<double>>  // x, y
plotEllipse(const Eigen::Matrix2d& cov, double mean_x, double mean_y) {
  // Compute the eigenvalues and eigenvectors
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cov);
  if (eigensolver.info() != Eigen::Success) {
    std::cerr << "Failed to compute eigenvalues and eigenvectors." << std::endl;
    return {};
  }

  // Eigenvalues are the lengths of the ellipse's axes
  Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
  double width = std::sqrt(eigenvalues(0)) * 4;
  double height = std::sqrt(eigenvalues(1)) * 4;

  // Eigenvectors are the directions of the ellipse's axes
  Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors();
  double angle = std::atan2(eigenvectors(1, 0), eigenvectors(0, 0));

  // Generate ellipse points
  std::vector<double> x, y;
  int num_points = 100;
  for (int i = 0; i < num_points; ++i) {
    double theta = 2.0 * M_PI * i / num_points;
    double x_ = width * std::cos(theta) / 2.0;
    double y_ = height * std::sin(theta) / 2.0;

    // Rotate the points
    double x_rot = std::cos(angle) * x_ - std::sin(angle) * y_;
    double y_rot = std::sin(angle) * x_ + std::cos(angle) * y_;

    // Translate the points
    x.push_back(x_rot + mean_x);
    y.push_back(y_rot + mean_y);
  }

  return {x, y};
}

inline void createWindow(sf::RenderWindow* window,
                         tgui::Gui* gui,
                         std::function<void()> onGtsam) {
  window->create(sf::VideoMode(800, 600), "GBPC Visualization");
  gui->setWindow(*window);

  auto child = tgui::ChildWindow::create();
  child->setClientSize({100, 120});
  child->setPosition(420, 80);
  child->setTitle("Options");
  gui->add(child);

  auto comboBox = tgui::ComboBox::create();
  comboBox->setSize(80, 20);
  comboBox->setPosition(10, 10);
  comboBox->addItem("Merge");
  comboBox->addItem("MergeRobust");
  comboBox->addItem("Average");
  comboBox->setSelectedItem("Merge");
  child->add(comboBox);

  // add a reset button
  auto reset_button = tgui::Button::create();
  reset_button->setPosition(10, 35);
  reset_button->setText("Reset");
  reset_button->setSize(80, 30);
  child->add(reset_button);

  // message box
  tgui::MessageBox::Ptr messageBox = tgui::MessageBox::create();
  messageBox->setText("not implemented");
  messageBox->addButton("OK");
  messageBox->onButtonPress([&] { messageBox->close(); });

  auto gtsam_opt_button = tgui::Button::create();
  gtsam_opt_button->setPosition(10, 70);
  gtsam_opt_button->setText("optimize with gtsam");
  gtsam_opt_button->setSize(80, 30);
  gtsam_opt_button->onPress([&] { onGtsam(); });
  child->add(gtsam_opt_button);
}

inline auto plotEllipse(sf::RenderWindow* window,
                        const sf::Color& color,
                        Gaussian gaussian) {
  auto mean = gaussian.mu();
  auto cov = gaussian.Sigma();
  double mean_x = mean[0], mean_y = mean[1];

  // Eigen decomposition of the covariance matrix
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cov);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("Failed to compute eigen decomposition.");
  }

  // Extract eigenvalues and eigenvectors
  Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
  Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors();

  // The eigenvalues are the squared lengths of the semi-major and semi-minor
  // axes
  double a = std::sqrt(eigenvalues(0));
  double b = std::sqrt(eigenvalues(1));

  // The eigenvector corresponding to the largest eigenvalue
  Eigen::Vector2d majorAxis = eigenvectors.col(0);

  // Calculate the rotation angle in degrees
  double angle = std::atan2(majorAxis(1), majorAxis(0)) * 180 / M_PI;

  // Create the ellipse as a circle and scale it
  sf::CircleShape ellipse(1.0f);        // Initial radius of 1
  ellipse.setScale(a * 2., b * 2.);     // Scale it to the correct size
  ellipse.setOrigin(1.0f, 1.0f);        // Set origin to center of the circle
  ellipse.setPosition(mean_x, mean_y);  // Set position to the mean
  ellipse.setRotation(angle);           // Rotate it to the correct orientation
  ellipse.setFillColor(color);          // Set the color

  window->draw(ellipse);

  return ellipse;
}

// Function to map a scalar value to an SFML color
inline sf::Color getColorFromValue(float value,
                                   float minValue,
                                   float maxValue) {
  // Normalize value to [0, 1]
  float normalized = (value - minValue) / (maxValue - minValue);

  // Clamp the normalized value to [0, 1]
  normalized = std::min(1.0f, std::max(0.0f, normalized));

  // Map the normalized value to a color
  sf::Uint8 red = static_cast<sf::Uint8>(normalized * 255);
  sf::Uint8 blue = static_cast<sf::Uint8>((1.0f - normalized) * 255);

  return sf::Color(red, 0, blue);
}

inline sf::Color setAlpha(sf::Color color, float alpha) {
  color.a = static_cast<sf::Uint8>(alpha * 255);
  return color;
}

inline void plotConnectBeliefs(sf::RenderWindow* window,
                               const sf::Color& color,
                               const Gaussian& gaussian1,
                               const Gaussian& gaussian2) {
  auto mean1 = gaussian1.mu();
  auto mean2 = gaussian2.mu();

  sf::Vertex line[] = {
      sf::Vertex(sf::Vector2f(mean1[0], mean1[1]), color),
      sf::Vertex(sf::Vector2f(mean2[0], mean2[1]), color),
  };

  window->draw(line, 2, sf::Lines);
}
}  // namespace gbpc::visualization

#endif  // GBPC_VISUALIZATION_VISUALIZATION_H_