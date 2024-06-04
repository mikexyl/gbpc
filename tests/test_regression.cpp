#include <iostream>
#include <random>

#include "gbpc/variable.h"

using namespace gbpc;

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());

  // test 1d regression
  float gt_mu = 0., gt_Sigma = 1.;
  std::normal_distribution<> d_1d(gt_mu, gt_Sigma);
  const int num_samples = 100;
  float initial_value = d_1d(gen);
  Gaussian<1> initial = Gaussian<1>::fromMuSigma(
      Eigen::Vector<double, 1>(initial_value),
      Eigen::Matrix<double, 1, 1>::Identity() * gt_Sigma);
  Variable<1> vn_1d(initial);

  std::cout << "1d gt_mu: " << gt_mu << std::endl;
  for (int i = 0; i < num_samples; i++) {
    float mu = d_1d(gen);
    Eigen::Vector<double, 1> mu_vec;
    mu_vec << mu;
    Eigen::Matrix<double, 1, 1> Sigma;
    Sigma << gt_Sigma;
    auto message = Gaussian<1>::fromMuSigma(mu_vec, Sigma);
    vn_1d.update({message});
  }

  std::cout << "1d regression: mu = " << vn_1d.mu()(0)
            << ", Sigma = " << vn_1d.sigma()(0, 0) << std::endl;
}