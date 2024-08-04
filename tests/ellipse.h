#pragma once

#include <Eigen/Eigen>
#include <QColor>
#include <QPainter>

class Ellipse {
 public:
  Ellipse(const Eigen::Vector2d& mean,
          const Eigen::Matrix2d& covariance,
          const Eigen::Vector3d& color = Eigen::Vector3d(0, 0, 0),
          bool draw_ellipse = true)
      : mean(mean),
        covariance(covariance),
        draw_ellipse(draw_ellipse),
        color(QColor(color[0], color[1], color[2], 50)) {
    calculateParameters();
  }

  void draw(QPainter& painter) const {
    painter.save();
    painter.translate(mean[0], mean[1]);

    if (draw_ellipse) {
      painter.setPen(Qt::red);
      painter.setBrush(color);  // Transparent red fill color
      painter.rotate(angle);
      painter.drawEllipse(QPointF(0, 0), axisLength1, axisLength2);
    }

    // draw mean dot
    painter.setPen(Qt::black);
    auto dot_color = this->color;
    dot_color.setAlpha(255);
    painter.setBrush(dot_color);
    painter.drawEllipse(QPointF(0, 0), 2, 2);

    painter.restore();
  }

  Eigen::Vector2d getMean() const { return mean; }

 private:
  Eigen::Vector2d mean;
  Eigen::Matrix2d covariance;
  double axisLength1;
  double axisLength2;
  double angle;
  bool draw_ellipse;
  QColor color;
  static QColor HighlightColor;

  void calculateParameters() {
    // Compute eigenvalues and eigenvectors of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
    Eigen::Vector2d eigenvalues = solver.eigenvalues();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors();

    // The lengths of the semi-axes are proportional to the square roots of the
    // eigenvalues
    axisLength1 =
        std::sqrt(eigenvalues[1]) * 2;  // Scale factor for visualization
    axisLength2 =
        std::sqrt(eigenvalues[0]) * 2;  // Scale factor for visualization

    // The angle of rotation of the ellipse
    angle = std::atan2(eigenvectors(1, 1), eigenvectors(0, 1)) * 180 / M_PI;
  }

 public:
  bool contains(const QPointF& point) const {
    // Inverse transform the point to the ellipse's coordinate system
    QTransform transform;
    transform.translate(mean[0], mean[1]);
    transform.rotate(angle);
    QPointF localPoint = transform.inverted().map(point);

    // Check if the point is within the ellipse using the ellipse equation
    double x = localPoint.x();
    double y = localPoint.y();
    return (x * x) / (axisLength1 * axisLength1) +
               (y * y) / (axisLength2 * axisLength2) <=
           1.0;
  }
};
