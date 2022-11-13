#pragma once

#include <iostream>
#include <string>

#include <glog/logging.h>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>

#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Cal3_S2.h>

namespace vdo {
namespace utils {

gtsam::Pose3 cvMatToGtsamPose3(const cv::Mat& H);
 // Converts a rotation matrix and translation vector from opencv to gtsam
  // pose3
gtsam::Pose3 cvMatsToGtsamPose3(const cv::Mat& R, const cv::Mat& T);

cv::Mat gtsamPose3ToCvMat(const gtsam::Pose3& pose);

/* ------------------------------------------------------------------------ */
// Converts a 3x3 rotation matrix from opencv to gtsam Rot3
gtsam::Rot3 cvMatToGtsamRot3(const cv::Mat& R);

} //utils
} //vdo