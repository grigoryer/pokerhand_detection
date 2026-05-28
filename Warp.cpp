#include "Warp.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& cornerPoints)
{
    // sort by y to get top two and bottom two
    std::vector<cv::Point2f> sorted = cornerPoints;

    std::sort(sorted.begin(), sorted.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

    std::vector<cv::Point2f> top = {sorted[0], sorted[1]};
    std::vector<cv::Point2f> bottom = {sorted[2], sorted[3]};

    // within each pair sort by x to get left/right
    std::sort(top.begin(), top.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    std::sort(bottom.begin(), bottom.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    sorted[0] = top[0];    // TL
    sorted[1] = top[1];     // TR
    sorted[2] = bottom[1]; // BR
    sorted[3] = bottom[0];  // BL
    
    return sorted;
}


// first is going to be the entire card, second is the corner extracted.
std::pair<std::vector<Mat>,std::vector<Mat>> extractCards(Mat& img, std::vector<std::vector<cv::Point2f>> cardsPoint)
{
    static const std::vector<cv::Point2f> destPoints = {{0, 0}, {(float)CARD_WIDTH, 0}, {(float)CARD_WIDTH, (float)CARD_HEIGHT}, {0, (float)CARD_HEIGHT}};
    std::pair<std::vector<Mat>,std::vector<Mat>>  warped_cards;

    for (const auto& approxPoints : cardsPoint)
    {
        auto orderedPoints = orderPoints(approxPoints);

        Mat warped;
        warpPerspective(img, warped, getPerspectiveTransform(orderedPoints, destPoints), cv::Size(CARD_WIDTH, CARD_HEIGHT));

        // If image is in wrong oreintation landscape instead portrait the symbol will be in right box, if so we flip.
        Mat topLeft = warped(cv::Rect(0, 0, cornerWidth, cornerHeight));
        Mat topRight = warped(cv::Rect(CARD_WIDTH - cornerWidth, 0, cornerWidth, cornerHeight));
        
        // if TL is brighter than TR, ink is on the right side, rotate 90
        if (cv::mean(topLeft)[0] > cv::mean(topRight)[0])
        {
            cv::rotate(warped, warped, cv::ROTATE_90_CLOCKWISE);
            cv::resize(warped, warped, cv::Size(CARD_WIDTH, CARD_HEIGHT));
        }
            
        Mat idCorner = warped(cv::Rect(0, 0, cornerWidth, CARD_HEIGHT / 3)).clone();

        warped_cards.first.push_back(warped);
        warped_cards.second.push_back(idCorner);

        // draw debug boxes
        cv::rectangle(warped, cv::Rect(0, 0, cornerWidth, cornerHeight), Scalar(0, 255, 0), 3);
        cv::rectangle(warped, cv::Rect(CARD_WIDTH - cornerWidth, 0, cornerWidth, cornerHeight), Scalar(0, 0, 255), 3);

        for (int i = 0; i < 4; i++)
        {
            line(img, orderedPoints[i], orderedPoints[(i + 1) % 4], Scalar(0, 255, 0), 3);
            circle(img, orderedPoints[i], 30, Scalar(0, 0, 255), 5);
            putText(img, std::to_string(i), orderedPoints[i] + cv::Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 3, Scalar(255, 255, 0), 10);

            double dist = cv::norm(orderedPoints[i] - orderedPoints[(i + 1) % 4]);
            cv::Point2f mid = (orderedPoints[i] + orderedPoints[(i + 1) % 4]) * 0.5f;
            putText(img, std::to_string((int)dist), mid + cv::Point2f(5, -5), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 100, 0), 5);
        }
    }
    return warped_cards;
}

std::vector<std::vector<cv::Point2f>> detectCardContours(Mat& img)
{
    // Loads an image and greyscale it.
    Mat src;
    cvtColor(img, src, COLOR_BGR2GRAY);
    threshold(src, src, 196, 255, THRESH_BINARY);

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