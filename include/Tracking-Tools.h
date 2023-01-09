#pragma once

#include "Macros.h"
#include "Types.h"
#include "Frame.h"

#include <gtsam/base/Vector.h>



namespace vdo {
class Camera;

gtsam::Vector3 obtainFlowDepth(const Frame& frame, TrackletId tracklet_id);
void determineOutlierIds(const TrackletIds& inliers, const TrackletIds& tracklets, TrackletIds& outliers);

void updateMotionModel(const Frame::Ptr& previous_frame, Frame::Ptr current_frame);
void calculateSceneFlow(const Frame::Ptr& previous_frame, Frame::Ptr current_frame, const Camera& camera);

//using the previous frame updates the mask of the upcoming frame
void updateFrameMask(const Frame::Ptr& previous_frame, Frame::Ptr current_frame);

}