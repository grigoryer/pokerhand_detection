#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
 
using namespace cv;

std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& rotatedPoints, const std::vector<cv::Point2f>& cornerPoints);

std::vector<Mat> extractCards(Mat& img, std::vector<std::vector<cv::Point2f>> cardsPoint);

std::vector<std::vector<cv::Point2f>> detectCardContours(Mat& img);
