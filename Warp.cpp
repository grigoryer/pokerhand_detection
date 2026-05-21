#include "Warp.hpp"
#include <cmath>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

const int CARD_WIDTH  = 610;
const int CARD_HEIGHT = 860;

using namespace cv;

std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& rotatedPoints, const std::vector<cv::Point2f>& cornerPoints)
{
    std::vector<cv::Point2f> ordered(4);
    
    // find each corner by sum and difference
    float minSum = FLT_MAX, maxSum = -FLT_MAX;
    float minDiff = FLT_MAX, maxDiff = -FLT_MAX;
    
    for (int i = 0; i < 4; i++)
    {
        auto p = rotatedPoints[i];
        float sum  = p.x + p.y;
        float diff = p.x - p.y;
    
        if (sum  < minSum)  { minSum = sum; ordered[0] = cornerPoints[i]; } // top-left
        if (sum  > maxSum)  { maxSum = sum; ordered[2] = cornerPoints[i]; } // bottom-right
        if (diff < minDiff) { minDiff = diff; ordered[3] = cornerPoints[i]; } // bottom-left
        if (diff > maxDiff) { maxDiff = diff; ordered[1] = cornerPoints[i]; } // top-right
    }

    return ordered;
}

std::vector<Mat> extractCards(Mat& img, std::vector<std::vector<cv::Point2f>> cardsPoint)
{
    static const std::vector<cv::Point2f> destPoints = {{0, 0}, {(float)CARD_WIDTH, 0}, {(float)CARD_WIDTH, (float)CARD_HEIGHT}, {0, (float)CARD_HEIGHT}};
    std::vector<Mat> warped_cards;
    
    for (const auto& approxPoints : cardsPoint)
    {
        // We need to make sure that the cards enter the orderCards where they are portrait.
        // All ordered does is make sure that order is correct for the perspective Transform.

        auto rect = minAreaRect(approxPoints);
        float angle = rect.angle;

        if (rect.size.width > rect.size.height) { angle += 90.0f; }

        // Create rotational Matrix we now have 1 to 1 map of rotated and current points, we use rotation in calculation but move the real points
        std::vector<cv::Point2f> rotatedPoints;
        transform(approxPoints, rotatedPoints, getRotationMatrix2D(rect.center, angle, 1.0));

        auto orderedPoints = orderPoints(rotatedPoints, approxPoints);

        // Draw lines to identify what we are scanning
        for (int i = 0; i < 4; i++)
        {
            line(img, approxPoints[i], approxPoints[(i + 1) % 4], Scalar(0, 255, 0), 3);
        }

        Mat warped;
        warpPerspective(img, warped, getPerspectiveTransform(orderedPoints, destPoints), cv::Size(CARD_WIDTH, CARD_HEIGHT));
        
        warped_cards.push_back(warped);
    }

    return warped_cards;
}


std::vector<std::vector<cv::Point2f>> detectCardContours(Mat& img)
{
    // Loads an image and greyscale it.
    Mat src;
    cvtColor(img, src, COLOR_BGR2GRAY);
    threshold(src, src, 196, 255, 0);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    //Finds all contours
    
    findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point2f>> cardsPoint;

    for (const auto& cnt : contours)
    {
        // arbitrary size  for now, works well and deletes tiny specs 
        // also ensures cards are large enough for identifitication
        if (contourArea(cnt) < 6000) { continue; }
        
        // Idenfity conotur rectangle and get points of edges
        std::vector<cv::Point2f> approxPoints;
        approxPolyDP(cnt, approxPoints, (0.02 * arcLength(cnt, true)), true);

        // If shape isnt a rectangle with 4 points or concave we can skip.
        if (approxPoints.size() != 4 || (!isContourConvex(approxPoints))) { continue; }

        cardsPoint.push_back(approxPoints);
    }

    return cardsPoint; // no card found
}