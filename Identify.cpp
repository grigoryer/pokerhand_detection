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

static auto orb = cv::ORB::create(80);

static std::vector<cv::Mat> rankDescriptors;
static std::vector<std::vector<cv::KeyPoint>> rankKeypoints;


Mat tangentHistogram(const vector<Point>& contour)
{
    int bins = 180;

    Mat hist = Mat::zeros(1, bins, CV_32F);
    int n = contour.size();
    
    // for each line find the tangent line to the edge, add it to bin.
    for (int i = 0; i < n; i++)
    {
        Point prev = contour[(i - 1 + n) % n];
        Point next = contour[(i + 1) % n];
        float angle = atan2f(next.y - prev.y, next.x - prev.x);
        angle += CV_PI;
        int bin = (int)(angle / (2 * CV_PI) * bins) % bins;
        hist.at<float>(0, bin)++;
    }

    normalize(hist, hist, 1, 0, NORM_L1);
    return hist;
}

std::vector<std::vector<cv::Point>> findSortedContours(Mat& img, const int cannyLow, const int cannyHigh)
{
    Mat grey, edges;

    //Convert to gray and preprocess threshold.
    cvtColor(img, grey, COLOR_BGR2GRAY);
    threshold(grey, grey, 0, 255, THRESH_BINARY | THRESH_OTSU);
    
    // Find edges and increase edge thickness.
    Canny(grey, edges, cannyLow, cannyHigh);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(edges, edges, kernel, Point(-1, -1), 1);

    // Find contours, use "CHAIN_APPROX_NONE" for more exact cords. 
    // Sort based on area and only use the biggest contour, prevents using a random dot instead.
    std::vector<std::vector<cv::Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    std::sort(contours.begin(), contours.end(), [](const auto& a, const auto& b)
        { return contourArea(a) > contourArea(b); });

    return contours;
}

void initTemplates()
{
    // Histograms for suit
    suitHistograms.resize(4);
    for (int i{}; i < 4; i++)
    {
        Mat suitTemplate = imread("cards/suit/" + to_string(i) + ".png");

        // Inrease make the tangent lines more visisble
        cv::resize(suitTemplate, suitTemplate, Size(250,300));

        // Preprocesses template Image for histogram
        cv::GaussianBlur(suitTemplate, suitTemplate, Size(3,1), 0);
        cv::GaussianBlur(suitTemplate, suitTemplate, Size(1,3), 0);

        // Find edges and increase edge thickness.
        auto contours = findSortedContours(suitTemplate, 10, 80);
        suitHistograms[i] = tangentHistogram(contours[0]);
    }

    // Orb for rank
    rankDescriptors.resize(13);
    rankKeypoints.resize(13);

    for (int i = 0; i < 13; i++)
    {
        Mat templ = imread("cards/rank/" + to_string(i) + ".png", IMREAD_GRAYSCALE);
        threshold(templ, templ, 0, 255, THRESH_BINARY | THRESH_OTSU);
        cv::resize(templ, templ, Size(250, 300));
        orb->detectAndCompute(templ, cv::noArray(), rankKeypoints[i], rankDescriptors[i]);
    }
}

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

String numToRank(int rank)
{
    switch (rank)
    {
        case (0) : return "Ace";
        case (1) : return "Two";
        case (2) : return "Three";
        case (3) : return "Four";
        case (4) : return "Five";
        case (5) : return "Six";
        case (6) : return "Seven";
        case (7) : return "Eight";
        case (8) : return "Nine";
        case (9) : return "Ten";
        case (10) : return "Jack";
        case (11) : return "Queen";
        case (12) : return "King";
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

    // Ratio of 15% or more needed to dominate, else it is unknown color.
    int total = redCount + blackCount;
    if (total == 0 || (double)std::abs(redCount - blackCount) / total < 0.15)
    {
        return COLOR::UNKNOWN;
    }

    // Return which count is larger
    if (redCount > blackCount) 
    {   
        return COLOR::RED;
    }

    return COLOR::BLACK;
}

SUIT identifySuit(Mat& imgCorner, COLOR suitColor)
{   
    // Find contours
    auto contours = findSortedContours(imgCorner, 10, 80);
    
    if (contours.empty())
    {
        std::cout << "No contours found, defaulting\n";
        return suitColor == COLOR::RED ? SUIT::HEART : SUIT::CLUB;
    }

    // Based on color check only needed colors. 
    int curSuit;
    int endSuit = 0;
    if (suitColor == COLOR::BLACK)
    {
        curSuit = static_cast<int>(SUIT::CLUB);
        endSuit = static_cast<int>(SUIT::SPADE);
    }
    else if (suitColor == COLOR::RED)
    {
        curSuit = static_cast<int>(SUIT::HEART);
        endSuit = static_cast<int>(SUIT::DIAMOND);
    }
    else 
    {
        curSuit = static_cast<int>(SUIT::HEART);
        endSuit = static_cast<int>(SUIT::SPADE);
    }
    
    // For the top 3 contours check relative scores, and keep the largest score as the best suit
    int bestSuit = curSuit;
    double bestScore = -1e9; 
    double score = bestScore;
    
    for (; curSuit <= endSuit; curSuit++)
    {
        score = -1e9;
        // double check for multiple contours. Go through top 3 based on area.
        for (int i = 0; i < std::min((int)contours.size(), 3); i++)
        {
            Mat curHist = tangentHistogram(contours[i]);
            double cntScore = compareHist(curHist, suitHistograms[curSuit], HISTCMP_CORREL);
            score = std::max(score, cntScore);
            
            // Debug draw contours and each of there scores.
            drawContours(imgCorner, contours, i, Scalar(0,255,0), 5);
            std::cout << numToSuit(curSuit) << " Score: " << cntScore << "\n";
        }

        std::cout << "Best Suit Score" << numToSuit(curSuit) << " " << score << "\n";
        
        if (score > bestScore)
        {
            bestSuit = curSuit;
            bestScore = score;
        }
    }   
    
    return static_cast<SUIT>(bestSuit);
}

RANK identifyRank(Mat& imgCorner)
{
    Mat grey;
    cvtColor(imgCorner, grey, COLOR_BGR2GRAY);
    threshold(grey, grey, 0, 255, THRESH_BINARY | THRESH_OTSU);
    cv::resize(grey, grey, Size(250, 300));

    std::vector<cv::KeyPoint> rankKP;
    cv::Mat rankDesc;
    orb->detectAndCompute(grey, cv::noArray(), rankKP, rankDesc);

    if (rankDesc.empty()) return RANK::ACE;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    int bestRank = 0;
    double bestScore = -1;

    for (int i = 0; i < 13; i++)
    {
        
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(rankDesc, rankDescriptors[i], knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        for (const auto& m : knnMatches)
        {
            if (m.size() >= 2 && m[0].distance < .80f * m[1].distance)
                goodMatches.push_back(m[0]);
        }

        double score = goodMatches.size();

        std::cout << numToRank(i) << " good matches: " << score << "\n";

        Mat templ = imread("cards/rank/" + to_string(i) + ".png", IMREAD_GRAYSCALE);
        threshold(templ, templ, 0, 255, THRESH_BINARY | THRESH_OTSU);
        cv::resize(templ, templ, Size(250, 300));
        cv::Mat result;
        cv::drawMatches(grey, rankKP, templ, rankKeypoints[i], goodMatches, result);
        cv::imshow("matches", result);
        waitKey(0);
        destroyWindow("matches");

        if (score > bestScore)
        {
            bestScore = score;
            bestRank = i;
        }
    }

    return static_cast<RANK>(bestRank);
}


CARD identifyCard(Mat& imgCorner, Mat& imgCard)
{  
    // Find card, 
    CARD cardType;

    // Find color
    COLOR suitColor = identifyColor(imgCorner);

    //Find rank and suit
    Mat rankCorner = imgCorner(cv::Rect(0, 0, cornerWidth, 130)).clone();
    imshow("rank", rankCorner);
    waitKey(0);
    destroyWindow("rank");

    cardType.first = identifyRank(rankCorner);
    cardType.second = identifySuit(imgCorner, suitColor);
    
    putText(imgCard, (numToRank((int)cardType.first)) + " of " + (numToSuit((int)cardType.second)), cv::Point2f(100, imgCard.size().height - 60), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 10);

    return cardType;
}