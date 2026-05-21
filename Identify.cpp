#include "Warp.hpp"
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

enum class SUIT { CLUB, DIAMOND, HEART, SPADE };

static std::vector<Mat> suitTemplates;

// FIRST IMPLEMENTATION USING TEMPLATES

void initTemplates()
{
    suitTemplates.resize(4);
    for (int i{}; i < 4; i++)
    {
        suitTemplates[i] = imread("cards/suit/" + to_string(i) + ".png");
        cv::resize(suitTemplates[i], suitTemplates[i], Size(76, 86));
    }
}


// First attempt, incredibly slow since we are checking entire image, 
// will improve later to only check conrners and compare that crop.

SUIT identitySuit(Mat& imgCard)
{
    int bestSuit = 0;
    float bestScore = -1.0f;
    
    for (int i{}; i < 4; i++)
    {
        Mat map;
        matchTemplate(imgCard, suitTemplates[i], map, TM_CCOEFF_NORMED);

        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(map, &minVal, &maxVal, &minLoc, &maxLoc);

        if (maxVal > bestScore)
        {
            bestSuit = i;
            bestScore = maxVal;
        }
    }
    
    String suit;
    switch (bestSuit)
    {
        case (0) : suit = "clubs"; break;  
        case (1) : suit = "diamond"; break;
        case (2) : suit = "hearts"; break;
        case (3) : suit = "spade"; break;
    }
    
    putText(imgCard, suit, cv::Point2f(100, imgCard.size().height - 60), FONT_HERSHEY_SIMPLEX, 3, Scalar(255, 0, 0), 10);

    return static_cast<SUIT>(bestSuit);
}


// SECOND IMPLMENTATION USING ORB keypoint comparison.

//auto orb = cv::ORB::create(30, 1.2f, 4, 0, 0, 2, ORB::FAST_SCORE, 10, 20);
