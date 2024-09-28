#ifndef GBPC_DEMO_UTILS_H_
#define GBPC_DEMO_UTILS_H_

#include <gsl/gsl_blas.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_vector.h>

#include <Eigen/Eigen>
#include <iostream>
#include <vector>

namespace gbpc {

struct data {
  size_t n;
  const std::vector<double>& x;
  const std::vector<double>& y;
};

inline int gaussian_f(const gsl_vector* params, void* data, gsl_vector* f) {
  size_t n = ((struct data*)data)->n;
  const std::vector<double>& x = ((struct data*)data)->x;
  const std::vector<double>& y = ((struct data*)data)->y;

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

inline auto fit_gaussian_2d(const std::vector<double>& x,
                            const std::vector<double>& y) {
  size_t n = x.size();

  struct data d = {n, x, y};

  const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;
  gsl_multifit_nlinear_workspace* work =
      gsl_multifit_nlinear_alloc(T, nullptr, n, 7);

  gsl_multifit_nlinear_fdf fdf;
  fdf.f = gaussian_f;
  fdf.df = nullptr;  // Use numerical differentiation for the Jacobian
  fdf.fvv = nullptr;
  fdf.n = n;
  fdf.p = 7;
  fdf.params = &d;

  gsl_vector* params = gsl_vector_alloc(7);
  gsl_vector_set(params, 0, 1.0);  // Initial guess for A
  gsl_vector_set(params, 1, 0.0);  // Initial guess for x0
  gsl_vector_set(params, 2, 0.0);  // Initial guess for y0
  gsl_vector_set(params, 3, 1.0);  // Initial guess for sigma_x
  gsl_vector_set(params, 4, 1.0);  // Initial guess for sigma_y
  gsl_vector_set(params, 5, 0.0);  // Initial guess for theta
  gsl_vector_set(params, 6, 0.0);  // Initial guess for offset

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
  gsl_matrix* covar = gsl_matrix_alloc(7, 7);
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

}  // namespace gbpc

#endif  // GBPC_DEMO_UTILS_H_