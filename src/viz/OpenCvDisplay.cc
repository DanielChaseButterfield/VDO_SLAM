#include "viz/OpenCvDisplay.h"
#include "utils/UtilsOpenCV.h"

#include "Frame.h"

#include <set>

namespace vdo
{
OpenCvDisplay::OpenCvDisplay(DisplayParams::Ptr params_) : Display(params_)
{
}

void OpenCvDisplay::process(const VisualiserInput& viz_input)
{
  if (!params->use_2d_viz)
  {
    return;
  }

  if (params->display_input)
  {
    drawInputImages(*viz_input.frontend_output->frame);
  }

  if (params->display_frame)
  {
    drawFrame(*viz_input.frontend_output->frame);
  }

  for (const ImageToDisplay& display : display_images)
  {
    cv::imshow(display.title, display.image);
  }

  cv::waitKey(1);
  display_images.clear();
}


void OpenCvDisplay::drawInputImages(const Frame& frame)
{
  // draw each portion of the inputs
  cv::Mat rgb, depth, flow, mask;
  frame.Images().rgb.copyTo(rgb);
  CHECK(rgb.channels() == 3) << "Expecting rgb in frame to gave 3 channels";

  frame.Images().depth.copyTo(depth);
  // expect depth in float 32
  depth.convertTo(depth, CV_8UC1);

  frame.Images().flow.copyTo(flow);

  frame.Images().semantic_mask.copyTo(mask);

  // canot display the original ones so these needs special treatment...
  cv::Mat flow_viz, mask_viz;
  drawOpticalFlow(flow, flow_viz);
  drawSemanticInstances(rgb, mask, mask_viz);

  cv::Mat top_row = utils::concatenateImagesHorizontally(rgb, depth);
  cv::Mat bottom_row = utils::concatenateImagesHorizontally(flow_viz, mask_viz);
  cv::Mat input_images = utils::concatenateImagesVertically(top_row, bottom_row);

  // reisize images to be the original image size
  cv::resize(input_images, input_images, cv::Size(rgb.cols, rgb.rows), 0, 0, CV_INTER_LINEAR);
  addDisplayImages(input_images, "Input Images");
}

void OpenCvDisplay::drawFrame(const Frame& frame)
{
  cv::Mat frame_viz, rgb;
  frame.Images().rgb.copyTo(rgb);
  CHECK(rgb.channels() == 3) << "Expecting rgb in frame to gave 3 channels";

  // drawTracklet(rgb, frame_viz, input.static_tracklets, input.map);

  drawFeatures(rgb, frame, frame_viz);
  addDisplayImages(frame_viz, "Current Frame");
}

void OpenCvDisplay::drawFeatures(const cv::Mat& rgb, const Frame& frame, cv::Mat& frame_viz)
{
  rgb.copyTo(frame_viz);
}

void OpenCvDisplay::drawOpticalFlow(const cv::Mat& flow, cv::Mat& flow_viz)
{
  CHECK(flow.channels() == 2) << "Expecting flow in frame to have 2 channels";

  // Visualization part
  cv::Mat flow_parts[2];
  cv::split(flow, flow_parts);

  // Convert the algorithm's output into Polar coordinates
  cv::Mat magnitude, angle, magn_norm;
  cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
  cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
  angle *= ((1.f / 360.f) * (180.f / 255.f));

  // Build hsv image
  cv::Mat _hsv[3], hsv, hsv8, bgr;
  _hsv[0] = angle;
  _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
  _hsv[2] = magn_norm;
  cv::merge(_hsv, 3, hsv);
  hsv.convertTo(hsv8, CV_8U, 255.0);

  // Display the results
  cv::cvtColor(hsv8, flow_viz, cv::COLOR_HSV2BGR);
}

void OpenCvDisplay::drawSemanticInstances(const cv::Mat& rgb, const cv::Mat& mask, cv::Mat& mask_viz)
{
  CHECK_EQ(rgb.size, mask.size) << "Input rgb and mask image must have the same size";
  rgb.copyTo(mask_viz);
  CHECK(mask.channels() == 1) << "Expecting mask input to have channels 1";
  CHECK(mask.depth() == CV_32SC1);

  for (int i = 0; i < mask.rows; i++)
  {
    for (int j = 0; j < mask.cols; j++)
    {
      // background is zero
      if (mask.at<int>(i, j) != 0)
      {
        Color color = getColourFromInstanceMask(mask.at<int>(i, j));
        // rgb or bgr?
        mask_viz.at<cv::Vec3b>(i, j)[0] = color.r;
        mask_viz.at<cv::Vec3b>(i, j)[1] = color.g;
        mask_viz.at<cv::Vec3b>(i, j)[2] = color.b;
      }
    }
  }
}

void OpenCvDisplay::addDisplayImages(const cv::Mat& image, const std::string& title)
{
  display_images.push_back(ImageToDisplay(image, title));
}

Color OpenCvDisplay::getColourFromInstanceMask(int value)
{
  return rainbowColorMap(value);
}

}  // namespace vdo