build:
  cmake --preset Release && \
  cmake --build --preset Release

build-gtsam:
  cd ../gtsam && \
  mkdir -p build && \
  cd build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_USE_SYSTEM_METIS=ON && \
  make -j4
