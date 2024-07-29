#include <gsl/gsl_blas.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_vector.h>

#include <SFML/Graphics.hpp>
#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>

#include "gbpc/gbpc.h"
#include "gbpc/visualization.h"
#include "utils.h"

using namespace gbpc;
using namespace gtsam;

auto generateSamples(int num_clusters, int max_samples, float distance) {
  std::vector<int> num_samples;
  std::vector<std::vector<double>> x, y;
  std::vector<Belief<Point2>> coverage;
  std::vector<float> centroids_x, centroids_y;
  std::vector<float> radii;

  // generate num_clusters centroids
  for (int i = 0; i < num_clusters; i++) {
    Eigen::Vector2d centroid;
    centroid << (i + 1) * distance, (i + 1) * distance;
    centroids_x.push_back(centroid(0));
    centroids_y.push_back(centroid(1));
    radii.push_back(rand() % static_cast<int>(distance * 0.3) +
                    static_cast<int>(distance * 0.2));
    num_samples.push_back(rand() % max_samples);
  }

  // generate num_samples points around each centroid
  for (int i = 0; i < num_clusters; i++) {
    x.push_back({});
    y.push_back({});
    for (int j = 0; j < num_samples[i]; j++) {
      Eigen::Vector2f sample_point;
      sample_point << centroids_x[i] + radii[i] * (rand() % 100) / 100.0,
          centroids_y[i] + radii[i] * (rand() % 100) / 100.0;
      x[i].push_back(sample_point(0));
      y[i].push_back(sample_point(1));
    }
  }

  // compute mean and std
  for (int i = 0; i < num_clusters; i++) {
    Eigen::Vector2d mean;
    mean << gsl_stats_mean(x[i].data(), 1, x[i].size()),
        gsl_stats_mean(y[i].data(), 1, y[i].size());
    Eigen::Matrix2d cov;
    cov << gsl_stats_variance(x[i].data(), 1, x[i].size()),
        gsl_stats_covariance(x[i].data(), 1, y[i].data(), 1, y[i].size()),
        gsl_stats_covariance(x[i].data(), 1, y[i].data(), 1, y[i].size()),
        gsl_stats_variance(y[i].data(), 1, y[i].size());
    coverage.push_back(Belief<Point2>(0, mean, cov, num_samples[i]));
  }

  return std::make_tuple(coverage, centroids_x, centroids_y, radii, x, y);
}

auto plotEllipse(sf::RenderWindow* window,
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

bool moveEllipse(sf::RenderWindow* window,
                 const sf::Color& color,
                 const Gaussian& gaussian,
                 const Point2& target_position,
                 float progress,
                 std::function<void()> finish_callback = {}) {
  auto start_position = gaussian.mu();
  auto curr_position =
      (1 - progress) * start_position + target_position * progress;

  Gaussian curr_gaussian(
      gaussian.key(), curr_position, gaussian.Sigma(), gaussian.N());

  plotEllipse(window, color, curr_gaussian);

  bool finished = progress >= 0.99;
  if (finished) {
    finish_callback();
  }

  return not finished;
}

void plotPoints(sf::RenderWindow* window,
                const sf::Color& color,
                const std::vector<double>& x,
                const std::vector<double>& y,
                float radius) {
  for (int i = 0; i < x.size(); i++) {
    sf::CircleShape point(radius);
    point.setFillColor(color);
    point.setPosition(x[i], y[i]);
    window->draw(point);
  }
}

void plotConnectBeliefs(sf::RenderWindow* window,
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

// Function to map a scalar value to an SFML color
sf::Color getColorFromValue(float value, float minValue, float maxValue) {
  // Normalize value to [0, 1]
  float normalized = (value - minValue) / (maxValue - minValue);

  // Clamp the normalized value to [0, 1]
  normalized = std::min(1.0f, std::max(0.0f, normalized));

  // Map the normalized value to a color
  sf::Uint8 red = static_cast<sf::Uint8>(normalized * 255);
  sf::Uint8 blue = static_cast<sf::Uint8>((1.0f - normalized) * 255);

  return sf::Color(red, 0, blue);
}

sf::Color setAlpha(sf::Color color, float alpha) {
  color.a = static_cast<sf::Uint8>(alpha * 255);
  return color;
}

int main() {
  static constexpr float kBeliefAlpha = 0.1;

  // set now time as seed
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  unsigned seed = now_ms.time_since_epoch().count();
  srand(seed);
  std::cout << "seed: " << seed;

  int window_width = 800, window_height = 800;
  int num_clusters = 3;
  auto const& [coverage, centroids_x, centroids_y, radii, x, y] =
      generateSamples(num_clusters, 100, window_height / 4.0);

  Graph graph;
  auto var = std::make_shared<Variable<Point2>>(coverage[0]);
  auto prior_factor = std::make_shared<gbpc::PriorFactor<Point2>>(var);
  graph.add(prior_factor);

  sf::RenderWindow window({static_cast<unsigned int>(window_width),
                           static_cast<unsigned int>(window_height)},
                          "GBP Animation");
  tgui::Gui gui(window);

  auto child = tgui::ChildWindow::create();
  child->setClientSize({100, 120});
  child->setPosition(420, 80);
  child->setTitle("Options");
  gui.add(child);

  auto comboBox = tgui::ComboBox::create();
  comboBox->setSize(80, 20);
  comboBox->setPosition(10, 10);
  comboBox->addItem("Merge");
  comboBox->addItem("MergeRobust");
  comboBox->addItem("Mixture");
  comboBox->setSelectedItem("Merge");
  child->add(comboBox);

  // add a reset button
  auto reset_button = tgui::Button::create();
  reset_button->setPosition(10, 35);
  reset_button->setText("Reset");
  reset_button->setSize(80, 30);
  reset_button->onPress([&var, &coverage] { var->setBelief(coverage[0]); });
  child->add(reset_button);

  // message box
  tgui::MessageBox::Ptr messageBox = tgui::MessageBox::create();
  messageBox->setText("not implemented");
  messageBox->addButton("OK");
  messageBox->onButtonPress([&] { messageBox->close(); });

  std::optional<Gaussian> gtsam_belief;

  auto gtsam_opt_button = tgui::Button::create();
  gtsam_opt_button->setPosition(10, 70);
  gtsam_opt_button->setText("optimize with gtsam");
  gtsam_opt_button->setSize(80, 30);
  gtsam_opt_button->onPress([&] {
    auto belief = Belief<Point2>::optimizeWithGtsam(coverage);
    gtsam_belief.emplace(belief);
    std::cout << *gtsam_belief << std::endl;
  });
  child->add(gtsam_opt_button);

  GaussianMergeType merge_type;
  auto get_merge_type = [&merge_type, &comboBox]() {
    auto selected_item = comboBox->getSelectedItem();
    if (selected_item == "Merge") {
      merge_type = GaussianMergeType::Merge;
    } else if (selected_item == "MergeRobust") {
      merge_type = GaussianMergeType::MergeRobust;
    } else if (selected_item == "Mixture") {
      merge_type = GaussianMergeType::Mixture;
    }
    return merge_type;
  };

  float freq = 30.0;

  std::vector<sf::Color> colors;
  for (int i = 0; i < 1000; i++) {
    colors.emplace_back(getColorFromValue(i, 0, num_clusters - 1));
  }

  sf::Clock clock;
  sf::Time animation_time =
      sf::seconds(0.5f);  // Set cooldown time to 0.5 seconds
  sf::Time animation_start_time = sf::Time::Zero;
  std::function<void()> finish_callback;

  int active_cluster = 0;
  bool animating = false;

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) window.close();
      gui.handleEvent(event);  // Pass the event to the GUI
    }
    window.clear(sf::Color::White);

    // Get elapsed time since the last key press
    sf::Time elapsed_time = clock.getElapsedTime() - animation_start_time;

    auto lambda_callback = [&prior_factor, &coverage](
                               int active_cluster,
                               GaussianMergeType merge_type) {
      prior_factor->update(coverage[active_cluster], merge_type);
    };

    // Check keyboard input with cooldown
    if (not animating) {
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
        // Perform the action for Space key
        active_cluster = (active_cluster - 1 + num_clusters) % num_clusters;

        // start animation
        auto merge_type = get_merge_type();
        finish_callback = [&lambda_callback, active_cluster, merge_type]() {
          lambda_callback(active_cluster, merge_type);
        };
        animating = true;
        animation_start_time = clock.getElapsedTime();
      } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q)) {
        // Perform the action for Q key
        window.close();
        exit(0);
      }
    } else {
      animating =
          moveEllipse(&window,
                      setAlpha(colors[active_cluster], kBeliefAlpha),
                      coverage[active_cluster],
                      graph.getNode<Point2>(0)->mu(),
                      elapsed_time.asSeconds() / animation_time.asSeconds(),
                      finish_callback);
    }

    if (gtsam_belief.has_value()) {
      plotEllipse(
          &window, setAlpha(sf::Color::Yellow, 0.6), gtsam_belief.value());
    }

    for (int i = 0; i < num_clusters; i++) {
      plotEllipse(&window, setAlpha(colors[i], 0.1), coverage[i]);
      plotPoints(&window, setAlpha(colors[i], 1.0), x[i], y[i], 1.);

      plotConnectBeliefs(
          &window, sf::Color::Black, coverage[i], *graph.getNode<Point2>(0));
    }

    plotEllipse(&window, setAlpha(colors[4], 0.6), *graph.getNode<Point2>(0));

    gui.draw();  // Draw the GUI
    window.display();

    sf::sleep(sf::milliseconds(1000 / freq));
  }

  return 0;
}
