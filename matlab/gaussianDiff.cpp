// mylibrary_mex.cpp
#include <iostream>

#include "gbpc/gbpc.h"
#include "mex.h"  // MATLAB MEX API

// Helper function to convert MATLAB matrix to Eigen::VectorXd
Eigen::VectorXd mxArrayToEigenVector(const mxArray* arr) {
  size_t rows = mxGetM(arr);
  size_t cols = mxGetN(arr);

  if (cols != 1) {
    mexErrMsgIdAndTxt("Gaussian:inputError", "Expected a column vector.");
  }

  Eigen::VectorXd vec(rows);
  double* data = mxGetPr(arr);
  for (size_t i = 0; i < rows; ++i) {
    vec(i) = data[i];
  }
  return vec;
}

// Helper function to convert MATLAB matrix to Eigen::MatrixXd
Eigen::MatrixXd mxArrayToEigenMatrix(const mxArray* arr) {
  size_t rows = mxGetM(arr);
  size_t cols = mxGetN(arr);

  Eigen::MatrixXd mat(rows, cols);
  double* data = mxGetPr(arr);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      mat(i, j) = data[i + j * rows];
    }
  }
  return mat;
}

// Main MEX function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  // Check the number of inputs
  if (nrhs != 4) {
    mexErrMsgIdAndTxt("Gaussian:invalidNumInputs",
                      "Four input arguments required.");
  }
  if (nlhs > 1) {
    mexErrMsgIdAndTxt("Gaussian:invalidNumOutputs",
                      "One output argument required.");
  }

  // Convert MATLAB inputs to Eigen types
  Eigen::VectorXd mu1 = mxArrayToEigenVector(prhs[0]);
  Eigen::VectorXd mu2 = mxArrayToEigenVector(prhs[1]);
  Eigen::MatrixXd cov1 = mxArrayToEigenMatrix(prhs[2]);
  Eigen::MatrixXd cov2 = mxArrayToEigenMatrix(prhs[3]);

  // Call the static method
  double result = gbpc::Gaussian::hellingerDistance(mu1, mu2, cov1, cov2);

  // Set the output
  plhs[0] = mxCreateDoubleScalar(result);
}
