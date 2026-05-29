#include <iostream>
#include <string>
#include <chrono>

#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui.hpp"

#include "Warp.hpp"
#include "Identify.hpp"

using namespace cv;
using namespace std;


void testCam()
{
    cv::VideoCapture cap(0, cv::CAP_AVFOUNDATION);

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    if (!cap.isOpened())
    {
        std::cerr << "Could not open camera\n";
        return;
    }

    // drain warm-up frames
    Mat frame;
    for (int i = 0; i < 30; i++) cap >> frame;
    int frameCount = 0;
    while (true)
    {
        cap >> frame;
        if (frame.empty()) continue;

        //std::cout << "frame: " << frame.cols << "x" << frame.rows 
                //<< " type=" << frame.type() << "\n";
                
        frameCount++;
        if (frame.empty()) continue;

        auto cords = detectCardContours(frame);
        auto cards = extractCards(frame, cords);
        
        imshow("card scanner", frame);
        
        if (!cards.first.empty())
        {
            auto card = identifyCard(cards.second[0], cards.first[0]);
            string name = "warped";
            imshow(name, cards.first[0]);
            imshow("warpedCorner", cards.second[0]);
        }

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
}

void testLocal()
{
    for (int i = 0; i < 8; i++)
    {
        std::cout << "\nNEW IMAGE " << i << "\n";
        string file = "build/images/card" + to_string(i) + ".jpeg";
        Mat img = imread(file);

        if (img.empty()) {
            cerr << "Could not load: " << file << "\n";
            continue;  // skip instead of crashing
        }

        auto startTime = std::chrono::high_resolution_clock::now();

        auto cardCords = detectCardContours(img);
        auto cards = extractCards(img, cardCords);

        size_t count = 0;

        for (size_t i = 0; i < cards.first.size(); i++)
        {
            identifyCard(cards.second[i], cards.first[i]);
            String name = "warped" + to_string(i);
            imshow(name, cards.first[i]);
            moveWindow(name, (150 * ((int)i + 1)), 0);
        }


        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsed = (endTime - startTime);

        std::cout << elapsed.count() << "\n";

        //imshow("card scanner", img);
        waitKey(0);
        destroyAllWindows();
    }
};


int main()
{
    initTemplates();
    testLocal();

    return 0;
}