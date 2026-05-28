#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
 
using namespace cv;


const int CARD_WIDTH  = 610;
const int CARD_HEIGHT = 860;


const int cornerHeight = CARD_HEIGHT / 3; //122
const int cornerWidth = CARD_WIDTH / 7; //86

std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& cornerPoints);

std::pair<std::vector<Mat>,std::vector<Mat>> extractCards(Mat& img, std::vector<std::vector<cv::Point2f>> cardsPoint);

std::vector<std::vector<cv::Point2f>> detectCardContours(Mat& img);
