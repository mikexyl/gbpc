#include <gsl/gsl_blas.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_vector.h>

#include <SFML/Graphics.hpp>

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
                 const Eigen::Matrix2d& cov,
                 const Eigen::Vector2d& mean) {
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
  ellipse.setFillColor(
      sf::Color(0, 255, 0, 100));  // Set the color with transparency

  window->draw(ellipse);

  return ellipse;
}

void plotPoints(sf::RenderWindow* window,
                const std::vector<double>& x,
                const std::vector<double>& y,
                float radius) {
  for (int i = 0; i < x.size(); i++) {
    sf::CircleShape point(radius);
    point.setFillColor(sf::Color::Red);
    point.setPosition(x[i], y[i]);
    window->draw(point);
  }
}

int main() {
  // set now time as seed
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  unsigned seed = now_ms.time_since_epoch().count();
  srand(seed);

  int window_width = 800, window_height = 800;
  int num_clusters = 3;
  auto const& [coverage, centroids_x, centroids_y, radii, x, y] =
      generateSamples(num_clusters, 100, window_height / 4.0);

  auto ellipse_xy = plotEllipse(
      coverage[0].Sigma(), coverage[0].mu()[0], coverage[0].mu()[1]);

  sf::RenderWindow window(sf::VideoMode(window_width, window_height),
                          "GBP Animation");
  sf::CircleShape point(5);
  point.setFillColor(sf::Color::Red);

  float freq = 30.0;

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) window.close();
    }

    window.clear();

    for (int i = 0; i < num_clusters; i++) {
      plotEllipse(&window, coverage[i].Sigma(), coverage[i].mu());
      plotPoints(&window, x[i], y[i], 1.);
    }

    window.draw(point);
    window.display();

    sf::sleep(sf::milliseconds(1000 / freq));
  }

  return 0;
}
