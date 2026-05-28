#include "Warp.hpp"
#include "Identify.hpp"
#include <functional>
#include <iostream>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

static std::vector<Mat> suitHistograms;
// FIRST IMPLEMENTATION USING TEMPLATES


Mat tangentHistogram(const vector<Point>& contour)
{
    int bins = 32;

    Mat hist = Mat::zeros(1, bins, CV_32F);
    int n = contour.size();
    
    for (int i = 0; i < n; i++)
    {
        Point prev = contour[(i - 1 + n) % n];
        Point next = contour[(i + 1) % n];
        float angle = atan2f(next.y - prev.y, next.x - prev.x); // -pi to pi
        angle += CV_PI; // shift to 0–2pi
        int bin = (int)(angle / (2 * CV_PI) * bins) % bins;
        hist.at<float>(0, bin)++;
    }
    
    normalize(hist, hist, 1, 0, NORM_L1);
    return hist;
}

struct SuitFeatures
{
    double centroidRatio; // 0 = top, 1 = bottom relative to bounding box
    double topMass;
    double bottomMass;
};

SuitFeatures extractFeatures(const std::vector<cv::Point>& contour)
{
    Moments m = moments(contour);
    double cy = m.m01 / m.m00;
    Rect bb = boundingRect(contour);
    double centroidRatio = (cy - bb.y) / bb.height;

    double midY = bb.y + bb.height / 2.0;
    int top = 0, bottom = 0;
    
    for (const auto& pt : contour)
    {
        if (pt.y < midY) { top++; }
        else { bottom++; }
    }

    double total = contour.size();
    return { centroidRatio, top / total, bottom / total };
}

void initTemplates()
{
    suitHistograms.resize(4);
    for (int i{}; i < 4; i++)
    {
        Mat suitTemplate = imread("cards/suit/" + to_string(i) + ".png", IMREAD_GRAYSCALE);
        Mat suitTemplateFull = imread("cards/suit/" + to_string(i) + ".png");

        cv::resize(suitTemplate, suitTemplate, Size(76, 86));
        GaussianBlur(suitTemplate, suitTemplate, Size(3,3), 0);
        

        std::vector<std::vector<cv::Point>> contours;
        Mat edges;
        Canny(suitTemplate, edges, 30, 200);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(edges, edges, kernel, Point(-1, -1), 1);
        
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        std::sort(contours.begin(), contours.end(), [](const auto& a, const auto& b)
            { return contourArea(a) > contourArea(b); });
            
        for (int j{}; j < contours.size(); j++)
        {
            drawContours(suitTemplateFull, contours, j, Scalar(255,0,0), 5);
        }
        
        // imshow("cnt", suitTemplateFull);
        // waitKey(0);
        // destroyWindow("cnt");
        suitHistograms[i] = tangentHistogram(contours[0]);
    }
}

// First attempt, incredibly slow since we are checking entire image, 
// will improve later to only check conrners and compare that crop.
String numToSuit(int suit)
{
    switch (suit)
    {
        case (0) : return "Hearts";
        case (1) : return "Diamond";
        case (2) : return "Clubs";
        case (3) : return "Spade";
    }

    return "error";
}

COLOR identifyColor(Mat& imgCorner)
{
    Mat hsvCorner;
    cvtColor(imgCorner, hsvCorner, COLOR_BGR2HSV);

    cv::Mat nonWhiteMask, redMaskLow, redMaskHigh, redMask, blackMask;
    // ignore near-white: V < 240 and S > 20
    cv::inRange(hsvCorner, cv::Scalar(0, 21, 0), cv::Scalar(179, 239, 239), nonWhiteMask);

    // red wraps: low and high hue ranges
    cv::inRange(hsvCorner, cv::Scalar(0, 51, 51), cv::Scalar(10, 255, 255), redMaskLow);
    cv::inRange(hsvCorner, cv::Scalar(170, 51, 51), cv::Scalar(179, 255, 255), redMaskHigh);
    cv::bitwise_or(redMaskLow, redMaskHigh, redMask);
    cv::bitwise_and(redMask, nonWhiteMask, redMask);

    // black: low V
    cv::inRange(hsvCorner, cv::Scalar(0, 0, 0), cv::Scalar(179, 80, 80), blackMask);
    cv::bitwise_and(blackMask, nonWhiteMask, blackMask);

    // counts
    int redCount = cv::countNonZero(redMask);
    int blackCount = cv::countNonZero(blackMask);

    std::cout << "\nRED COUNT: " << redCount << "\nBLACK COUNT: " << blackCount << "\n";

    return COLOR(redCount > blackCount);
}


SUIT identifySuit(Mat& imgCorner, COLOR suitColor)
{   
    std::vector<std::vector<cv::Point>> contours;
    Mat edges, grey;
    cvtColor(imgCorner, grey, COLOR_BGR2BGRA);
    Canny(grey, edges, 30, 200);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(edges, edges, kernel, Point(-1, -1), 1);
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        std::cout << "No contours found, defaulting\n";
        return suitColor == COLOR::RED ? SUIT::HEART : SUIT::CLUB;
    }

    std::sort(contours.begin(), contours.end(), [](const auto& a, const auto& b)
            { return contourArea(a) > contourArea(b); });



    //return early for red just use geometric difference.
    if (suitColor == COLOR::RED)
    {
        SuitFeatures f = extractFeatures(contours[0]);
        return (f.centroidRatio < 0.4 ? SUIT::HEART : SUIT::DIAMOND);
    }

    Mat curHist = tangentHistogram(contours[0]);
    double clubScore  = compareHist(curHist, suitHistograms[2], HISTCMP_CORREL);
    double spadeScore = compareHist(curHist, suitHistograms[3], HISTCMP_CORREL);

    std::cout << "Club: " << clubScore << " Spade: " << spadeScore << "\n";
    return clubScore > spadeScore ? SUIT::CLUB : SUIT::SPADE;

}


void identifyCard(Mat& imgCorner, Mat& imgCard)
{  
    Mat suitCorner = imgCorner(cv::Rect(0, 120, cornerWidth, (cornerHeight / 2))).clone();
    cv::GaussianBlur(suitCorner, suitCorner, Size(3,1), 10);
    cv::GaussianBlur(suitCorner, suitCorner, Size(1,3), 10);

    COLOR suitColor = identifyColor(suitCorner);
    SUIT suit = identifySuit(suitCorner, suitColor);

    putText(imgCard, numToSuit((int)suit), cv::Point2f(100, imgCard.size().height - 60), FONT_HERSHEY_SIMPLEX, 3, Scalar(255, 0, 0), 10);
}
