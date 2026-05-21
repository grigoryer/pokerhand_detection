#include <iostream>
#include <ratio>
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
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Could not open camera\n";
        return;
    }

    Mat frame;
    
    int frameCount = 0;
    while (true)
    {
        cap >> frame;
        frameCount++;
        if (frame.empty()) break;

        auto cords = detectCardContours(frame);
        auto cards = extractCards(frame, cords);
        
        imshow("card scanner", frame);
        
        if (!cards.empty())
        {
            auto suit = identitySuit(cards[0]);
            string name = "warped";
            imshow(name, cards[0]);
        }

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
}

void testLocal()
{
    for (int i = 0; i < 1; i++)
    {
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

        for (auto& card : cards)
        {
            auto suit = identitySuit(card);
            String name = "warped" + to_string(count++);
            imshow(name, card);
            moveWindow(name, (150 * count++), 0);
        }


        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsed = (endTime - startTime);

        std::cout << elapsed.count() << "\n";

        imshow("card scanner", img);
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