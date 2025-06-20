#include <gtsam/geometry/Pose3.h>
#include <matplot/matplot.h>

#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <vector>

#include "gbpc/contraction/hellinger.h"
#include "gbpc/variable.h"

using namespace std;
using namespace gtsam;
using namespace gbpc;

// Helper to interpolate between two Pose3s by fraction t in [0,1]
Pose3 interpolatePose(const Pose3& start, const Pose3& end, double t) {
  Vector6 delta = Pose3::Logmap(start.inverse() * end);
  Vector6 interp_delta = t * delta;
  return start * Pose3::Expmap(interp_delta);
}

// Helper to linearly interpolate covariance matrices
Eigen::Matrix<double, 6, 6> interpolateCov(
    const Eigen::Matrix<double, 6, 6>& start,
    const Eigen::Matrix<double, 6, 6>& end,
    double t) {
  return (1.0 - t) * start + t * end;
}

// Plot a 2D Gaussian trajectory as ellipses and path
void plot_gaussian_trajectory_2d(
    const std::vector<gtsam::Pose3>& means,
    const std::vector<Eigen::Matrix<double, 6, 6>>& covs,
    const std::string& title_str = "2D Trajectory of Gaussians (Ellipses)",
    const std::string& cmap_name = "jet") {
  using namespace matplot;
  std::vector<double> xs, ys;
  double min_x = std::numeric_limits<double>::max(),
         max_x = std::numeric_limits<double>::lowest();
  double min_y = std::numeric_limits<double>::max(),
         max_y = std::numeric_limits<double>::lowest();
  for (size_t i = 0; i < means.size(); ++i) {
    double x = means[i].x();
    double y = means[i].y();
    xs.push_back(x);
    ys.push_back(y);
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  }
  plot(xs, ys, "-o");
  title(title_str);
  xlabel("x");
  ylabel("y");
  grid(on);
  axis("equal");
  // Set and use colormap for ellipses
  if (cmap_name == "autumn")
    colormap(palette::autumn(means.size()));
  else if (cmap_name == "jet")
    colormap(palette::jet(means.size()));
  else
    colormap(palette::viridis(means.size()));
  auto cmap = colormap();
  for (size_t i = 0; i < means.size(); ++i) {
    double x = xs[i];
    double y = ys[i];
    Eigen::Matrix2d cov2d = covs[i].block<2, 2>(3, 3);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov2d);
    if (eig.info() == Eigen::Success) {
      auto eigvals = eig.eigenvalues();
      double width = 2 * std::sqrt(eigvals(0));
      double height = 2 * std::sqrt(eigvals(1));
      auto h = matplot::ellipse(x - width / 2, y - height / 2, width, height);
      std::array<float, 3> color = {static_cast<float>(cmap[i][0]),
                                    static_cast<float>(cmap[i][1]),
                                    static_cast<float>(cmap[i][2])};
      h->color(color);
      h->line_width(2.0);
    }
  }
  matplot::xlim({min_x - 2, max_x + 2});
  matplot::ylim({min_y - 2, max_y + 2});
}

int main() {
  const size_t N = 20;  // number of interpolation steps

  // Initial pose and covariance
  Pose3 mu0 = Pose3::Identity();
  Eigen::Matrix<double, 6, 6> cov0 =
      Eigen::Matrix<double, 6, 6>::Identity() * 0.5;

  // Target pose (translation + rotation)
  Pose3 muT = Pose3(gtsam::Rot3::RzRyRx(0.9, -0.3, 0.9),
                    gtsam::Point3(10.0, 7.5, -5.2));
  Eigen::Matrix<double, 6, 6> covT =
      3 * Eigen::Matrix<double, 6, 6>::Identity();

  // Generate trajectory
  std::vector<Pose3> means{mu0};
  std::vector<Eigen::Matrix<double, 6, 6>> covs{cov0};
  // set random seed for reproducibility
  srand(42);
  for (size_t i = 1; i < N; ++i) {
    // randomize step
    auto step = 1 + (static_cast<double>(rand()) / RAND_MAX - 0.5);
    step *= 2;
    step *= 0.1;
    auto mu_p = means.back();
    auto cov_p = covs.back();
    // distance to target pose
    auto dmu = Pose3::Logmap(mu_p.inverse() * muT);
    auto mu_c = mu_p * Pose3::Expmap(dmu * step);
    auto cov_c = cov_p + (covT - cov_p) * step;

    means.push_back(mu_c);
    covs.push_back(cov_c);
  }

  // set random seed for reproducibility
  srand(42);
  std::vector<Pose3> contracted_means;
  std::vector<Eigen::Matrix<double, 6, 6>> contracted_covs;
  gbpc::Variable<Pose3> var(0, Pose3::Logmap(mu0), cov0, 1);
  gbpc::UpdateParams params{.type = gbpc::GaussianMergeType::Contract,
                            .d_reset = 0.9,
                            .contract_alpha = 0.9,
                            .metric_type = gbpc::MetricType::Hellinger};
  contracted_means.push_back(Pose3::Expmap(var.mu()));
  contracted_covs.push_back(var.Sigma());
  for (size_t i = 1; i < N; ++i) {
    // randomize step
    auto step = 1 + (static_cast<double>(rand()) / RAND_MAX - 0.5);
    step *= 2;
    auto mu_p = Pose3::Expmap(var.mu());
    auto cov_p = var.Sigma();
    // distance to target pose
    auto dmu = Pose3::Logmap(mu_p.inverse() * muT);
    auto mu_c = mu_p * Pose3::Expmap(dmu * step);
    auto cov_c = cov_p + (covT - cov_p) * step;
    gbpc::Variable<Pose3> var_c(0, Pose3::Logmap(mu_c), cov_c, 1);
    var.contract(var_c, params);

    contracted_means.push_back(Pose3::Expmap(var.mu()));
    contracted_covs.push_back(var.Sigma());
  }

  // Compute metrics
  std::vector<double> hell, alpha, beta;
  Hellinger::computeMetrics(
      contracted_means, contracted_covs, hell, alpha, beta);

  // Prepare x-axis
  std::vector<size_t> steps(hell.size());
  std::iota(steps.begin(), steps.end(), 0);

  // Plot using Matplot++
  using namespace matplot;
  auto f = figure(true);
  f->size(1200, 900);

  // Metrics subplots
  subplot(4, 1, 1);
  plot(steps, hell, "-o");
  title("Hellinger Distance");
  ylabel("Hellinger");
  grid(on);

  subplot(4, 1, 2);
  plot(steps, alpha, "-o");
  title("Alpha");
  ylabel("Alpha");
  grid(on);

  subplot(4, 1, 3);
  plot(steps, beta, "-o");
  title("Beta");
  xlabel("Step");
  ylabel("Beta");
  grid(on);

  // 2D ellipses subplot
  subplot(4, 1, 4);
  plot_gaussian_trajectory_2d(contracted_means, contracted_covs);
  matplot::xlim({-2, 12});
  matplot::ylim({-2, 10});
  // equal axis
  axis("equal");

  show();
  // Optionally: save("metrics_plot.png");

  return 0;
}
