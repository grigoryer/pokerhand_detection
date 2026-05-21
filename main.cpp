#include <iostream>
#include "Warp.hpp"
#include <string>

#include "opencv2/highgui.hpp"



using namespace cv;
using namespace std;

int main()
{
    /*cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Could not open camera\n";
        return -1;
    }

    Mat frame;
    
    int frameCount = 0;
    while (true)
    {
        cap >> frame;
        frameCount++;
        if (frameCount % 2 != 0) { continue; } // only process every 5th frame
        if (frame.empty()) break;

        auto cords = detectCardContours(frame);
        auto cards = extractCards(frame, cords);
        
        imshow("card scanner", frame);
        
        if (!cards.empty())
        {

            string name = "warped" + to_string(0);
            imshow(name, cards[0]);
        }

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();*/

    for (int i = 1; i < 10; i++)
    {
        string file = "build/images/card" + to_string(i) + ".jpeg";
        Mat img = imread(file);

        if (img.empty()) {
            cerr << "Could not load: " << file << "\n";
            continue;  // skip instead of crashing
        }

        auto cardCords = detectCardContours(img);
        auto cards = extractCards(img, cardCords);

        imshow("card scanner", img);

        for (int j = 0; j < (int)cards.size(); j++)
        {
            imshow("warped" + to_string(j), cards[j]);
            moveWindow("warped", j * 600, 0);
        }

        waitKey(0);
        destroyAllWindows();
    }

    return 0;
}