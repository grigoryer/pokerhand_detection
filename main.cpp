#include <iostream>
#include "Warp.hpp"
#include <string>
#include <cmath>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


using namespace cv;
using namespace std;

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Could not open camera\n";
        return -1;
    }

    Mat frame;
    while (true)
    {
        cap >> frame;
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
    destroyAllWindows();
    return 0;
}