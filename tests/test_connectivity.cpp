// file: example_plot.cpp
#include <Eigen/Eigen>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_vector.h>
#include <matplot/matplot.h>

#include "gbpc/gbpc.h"

using namespace matplot;
using namespace gbpc;

template <int Dim> class GroupOpsScalar {
public:
  using Vector = Eigen::Vector<double, Dim>;
  static Vector dx(const Vector &mu_0, const Vector &mu_1) {
    return mu_1 - mu_0;
  }

  static Vector toVector(double scalar) {
    return (Vector() << scalar).finished();
  }

  static double fromVector(const Vector &vec) { return vec(0); }
};

struct data {
  size_t n;
  const std::vector<double> &x;
  const std::vector<double> &y;
};

int gaussian_f(const gsl_vector *params, void *data, gsl_vector *f) {
  size_t n = ((struct data *)data)->n;
  const std::vector<double> &x = ((struct data *)data)->x;
  const std::vector<double> &y = ((struct data *)data)->y;

  double A = gsl_vector_get(params, 0);
  double x0 = gsl_vector_get(params, 1);
  double y0 = gsl_vector_get(params, 2);
  double sigma_x = gsl_vector_get(params, 3);
  double sigma_y = gsl_vector_get(params, 4);
  double theta = gsl_vector_get(params, 5);
  double offset = gsl_vector_get(params, 6);

  for (size_t i = 0; i < n; ++i) {
    double dx = x[i] - x0;
    double dy = y[i] - y0;
    double a = (cos(theta) * dx + sin(theta) * dy) / sigma_x;
    double b = (-sin(theta) * dx + cos(theta) * dy) / sigma_y;
    double fit = A * exp(-0.5 * (a * a + b * b)) + offset;
    gsl_vector_set(f, i, fit - y[i]);
  }

  return GSL_SUCCESS;
}

auto fit_gaussian_2d(const std::vector<double> &x,
                     const std::vector<double> &y) {
  size_t n = x.size();

  struct data d = {n, x, y};

  const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
  gsl_multifit_nlinear_workspace *work =
      gsl_multifit_nlinear_alloc(T, nullptr, n, 7);

  gsl_multifit_nlinear_fdf fdf;
  fdf.f = gaussian_f;
  fdf.df = nullptr; // Use numerical differentiation for the Jacobian
  fdf.fvv = nullptr;
  fdf.n = n;
  fdf.p = 7;
  fdf.params = &d;

  gsl_vector *params = gsl_vector_alloc(7);
  gsl_vector_set(params, 0, 1.0); // Initial guess for A
  gsl_vector_set(params, 1, 0.0); // Initial guess for x0
  gsl_vector_set(params, 2, 0.0); // Initial guess for y0
  gsl_vector_set(params, 3, 1.0); // Initial guess for sigma_x
  gsl_vector_set(params, 4, 1.0); // Initial guess for sigma_y
  gsl_vector_set(params, 5, 0.0); // Initial guess for theta
  gsl_vector_set(params, 6, 0.0); // Initial guess for offset

  gsl_multifit_nlinear_init(params, &fdf, work);

  int status;
  size_t iter = 0;
  const size_t max_iter = 200;
  double xtol = 1e-8, gtol = 1e-8, ftol = 1e-8;
  int info;

  do {
    status = gsl_multifit_nlinear_iterate(work);
    if (status) {
      break;
    }

    status = gsl_multifit_nlinear_test(xtol, gtol, ftol, &info, work);
    ++iter;
  } while (status == GSL_CONTINUE && iter < max_iter);

  if (status == GSL_SUCCESS) {
    std::cout << "Converged:\n";
  } else {
    std::cout << "Failed to converge:\n";
  }

  // Extract the covariance matrix
  gsl_matrix *covar = gsl_matrix_alloc(7, 7);
  gsl_multifit_nlinear_covar(work->J, 0.0, covar);

  // Extract the parameters
  Eigen::Vector2d mean;
  mean << gsl_vector_get(work->x, 1), gsl_vector_get(work->x, 2);
  Eigen::Matrix2d cov;
  cov << gsl_matrix_get(covar, 1, 1), gsl_matrix_get(covar, 1, 2),
      gsl_matrix_get(covar, 2, 1), gsl_matrix_get(covar, 2, 2);

  gsl_matrix_free(covar);
  gsl_vector_free(params);
  gsl_multifit_nlinear_free(work);

  return std::make_pair(mean, cov);
}

void plot_ellipse(const Eigen::Matrix2d &cov, double mean_x, double mean_y) {
  using namespace matplot;

  // Compute the eigenvalues and eigenvectors
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cov);
  if (eigensolver.info() != Eigen::Success) {
    std::cerr << "Failed to compute eigenvalues and eigenvectors." << std::endl;
    return;
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

  plot(x, y);
}

int main() {

  Graph<2, GroupOpsScalar<2>> coverage_graph;

  int num_clusters = 3;
  std::vector<int> num_samples;
  std::vector<std::vector<double>> x, y;
  std::vector<Gaussian<2>> coverage;
  std::vector<float> centroids_x, centroids_y;
  std::vector<float> radii;

  // generate num_clusters centroids
  for (int i = 0; i < num_clusters; i++) {
    Eigen::Vector2d centroid;
    centroid << i * 5.0, i * 5.0;
    centroids_x.push_back(centroid(0));
    centroids_y.push_back(centroid(1));
    radii.push_back(rand() % 3 + 1);
    num_samples.push_back(rand() % 100);
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
    coverage.push_back(Gaussian<2>::fromMuSigma(mean, cov, num_samples[i]));
  }

  // send to gbp graph
  for (int i = 0; i < num_clusters; i++) {
    auto gaussian = coverage[i];
    if (i == 0) {
      coverage_graph.addNode(0, gaussian, false, gaussian.Sigma_.diagonal());
    } else {
      // print hellinger distance
      std::cout << "Hellinger distance: "
                << coverage_graph.at(0)->adj_var()->belief().hellingerDistance(
                       gaussian)
                << std::endl;

      std::cout << coverage_graph.sendMessage(0, gaussian,
                                              GaussianMergeType::Mixture)
                << std::endl;
    }
  }

  std::cout << "hellinger distance: " << std::endl;
  std::cout << " 0 -> 0: " << coverage[0].hellingerDistance(coverage[0])
            << std::endl;
  std::cout << " 0 -> 1: " << coverage[0].hellingerDistance(coverage[1])
            << std::endl;
  std::cout << " 0 -> 2: " << coverage[0].hellingerDistance(coverage[2])
            << std::endl;
  std::cout << " 1 -> 2: " << coverage[1].hellingerDistance(coverage[2])
            << std::endl;
  std::cout << " x -> 0: "
            << coverage_graph.at(0)->adj_var()->belief().hellingerDistance(
                   coverage[0])
            << std::endl;
  std::cout << " x -> 1: "
            << coverage_graph.at(0)->adj_var()->belief().hellingerDistance(
                   coverage[1])
            << std::endl;
  std::cout << " x -> 2: "
            << coverage_graph.at(0)->adj_var()->belief().hellingerDistance(
                   coverage[2])
            << std::endl;

  CBGraph<2, GroupOpsScalar<2>> cb_graph;
  Gaussian<2>::Mu dummy_mu(0.0, 0.0);
  Gaussian<2>::Sigma dummy_sigma;
  dummy_sigma << 1.0, 0.0, 0.0, 1.0;
  Gaussian<2> dummy_gaussian(dummy_mu, dummy_sigma, 0);

  for (int i = 0; i < num_clusters; i++) {
    cb_graph.addNode(i, dummy_gaussian, coverage[i], false);
  }

  // keep self looping for 10 iterations
  for (int i = 0; i < 10; i++) {
    int j = 0;
    std::cout << cb_graph.sendMessage(j, dummy_gaussian, coverage[j])
              << std::endl;
  }

  // keep sending messages to 0
  for (int it = 0; it < 10; it++) {
    for (int i = 1; i < num_clusters; i++) {
      std::cout << cb_graph.sendMessage(0, dummy_gaussian, coverage[i])
                << std::endl;
    }
  }

  // Create a figure
  auto fig = figure();

  // plot the samples, and color by cluster
  for (int i = 0; i < num_clusters; i++) {
    plot(x[i], y[i], "*");
    // hold plot
    hold(on);

    double mean_x = coverage[i].mu()(0);
    double mean_y = coverage[i].mu()(1);

    // plot the mean and std as ellipses
    plot_ellipse(coverage[i].Sigma_, mean_x, mean_y);
    hold(on);
  }

  // plot the graph belief
  auto mu = coverage_graph.at(0)->adj_var()->mu();
  plot_ellipse(coverage_graph.at(0)->adj_var()->sigma(), mu(0), mu(1));
  hold(on);

  mu = cb_graph.getNodeCoverage(0)->mu();
  plot_ellipse(cb_graph.getNodeCoverage(0)->sigma(), mu(0), mu(1));

  // Set plot title and axis labels
  title("Multiple Ellipses");
  xlabel("X Axis");
  ylabel("Y Axis");

  show();

  // Wait for user input to close
  std::cout << "Press Enter to close the plot..." << std::endl;
  std::cin.get();

  return 0;
}
