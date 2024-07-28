//
// Created by Mike Liu on 7/28/24.
//

#ifndef GBPC_VISUALIZATION_H
#define GBPC_VISUALIZATION_H

namespace gbpc {

    inline void plotEllipse(const Eigen::Matrix2d &cov, double mean_x, double mean_y,
                            std::string line_spec, float line_width,
                            std::string label = "") {
        using namespace matplot;

        // Compute the eigenvalues and eigenvectors
        Eigen::SelfAdjointEigenSolver <Eigen::Matrix2d> eigensolver(cov);
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

        auto p = plot(x, y, line_spec.c_str());
        p->line_width(line_width);
        // p->display_name(label);
    }
}

#endif //GBPC_VISUALIZATION_H
