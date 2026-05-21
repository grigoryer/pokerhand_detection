#include <iostream>
#include <string>
#include <cmath>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;


std::vector<cv::Point2f> orderPoints(std::vector<cv::Point>& rotatedPoints, std::vector<cv::Point>& cornerPoints)
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
    
        if (sum  < minSum)  { minSum  = sum;  ordered[0] = cornerPoints[i]; } // top-left
        if (sum  > maxSum)  { maxSum  = sum;  ordered[2] = cornerPoints[i]; } // bottom-right
        if (diff < minDiff) { minDiff = diff; ordered[3] = cornerPoints[i]; } // bottom-left
        if (diff > maxDiff) { maxDiff = diff; ordered[1] = cornerPoints[i]; } // top-right
    }

    float widthDist  = cv::norm(ordered[0] - ordered[1]); // tl -> tr
    float heightDist = cv::norm(ordered[0] - ordered[3]); // tl -> bl

    if (widthDist > heightDist) // landscape, cycle points 90°
    {
        std::vector<cv::Point2f> portrait = {
            ordered[3], // bl -> tl
            ordered[0], // tl -> tr
            ordered[1], // tr -> br
            ordered[2]  // br -> bl
        };
        return portrait;
    }

    return ordered;
}


std::vector<cv::Point> rotateContour(std::vector<cv::Point>& pts, float angleDeg)
{
    // Find centroid
    auto m = moments(pts);
    float cx = m.m10 / m.m00;
    float cy = m.m01 / m.m00;

    // Lambda helpers
    auto cart2pol = [](float x, float y, float& theta, float& rho)
    {
        theta = std::atan2(y, x);
        rho   = std::hypot(x, y);
    };
    auto pol2cart = [](float theta, float rho, float& x, float& y)
    {
        x = rho * std::cos(theta);
        y = rho * std::sin(theta);
    };

    // Normalize points to centroid
    std::vector<cv::Point2f> cnt_norm(pts.size());
    for (int i = 0; i < pts.size(); i++)
    {
        cnt_norm[i].x = pts[i].x - cx;
        cnt_norm[i].y = pts[i].y - cy;
    }

    // Convert to polar, rotate, convert back
    std::vector<cv::Point> cnt_rotated(pts.size());
    for (int i = 0; i < pts.size(); i++)
    {
        float theta, rho;
        cart2pol(cnt_norm[i].x, cnt_norm[i].y, theta, rho);

        // rad -> deg, add angle, wrap to [0, 360), deg -> rad
        float thetaDeg = theta * (180.0f / M_PI);
        thetaDeg = std::fmod(thetaDeg + angleDeg, 360.0f);
        theta = thetaDeg * (M_PI / 180.0f);

        float x, y;
        pol2cart(theta, rho, x, y);

        // Re-add centroid and store as integer Point
        cnt_rotated[i].x = std::round(x + cx);
        cnt_rotated[i].y = std::round(y + cy);
    }

    return cnt_rotated;
}


void getCardFromImage(const String file)
{
    // Loads an image and greyscale it.
    Mat src;
    Mat img = imread(samples::findFile(file));
    cvtColor(img, src, COLOR_BGR2GRAY);

    // Get threshold of image.
    threshold(src, src, 196, 255, 0);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& cnt : contours)
    {
        if (contourArea(cnt) > 10000) // arbitrary for now, works well
        {   
            //approx values of image.
            std::vector<cv::Point> approxPoints;
            auto perimeter = arcLength(cnt, true);
            approxPolyDP(cnt, approxPoints, 0.02 * perimeter, true);

            if (approxPoints.size() != 4 || (!isContourConvex(approxPoints))) continue;

            Mat lines = img.clone();
            for (int i = 0; i < 4; i++)
            {
                line(lines, approxPoints[i], approxPoints[(i + 1) % 4], Scalar(0, 255, 0), 7);
            }

            const int width = 700;
            const int height = 900;

            std::vector<cv::Point2f> destPoints = {{0, 0}, {width, 0}, {width, height}, {0, height}};
            
            auto rect = minAreaRect(approxPoints);
            auto rotatedPoints = rotateContour(approxPoints, std::fmod(rect.angle, 180.0f));
            

            for (int i = 0; i < 4; i++)
            {
                line(lines, rotatedPoints[i], rotatedPoints[(i + 1) % 4], Scalar(0,0,255), 7);
            }
            
            auto orderedPoints = orderPoints(rotatedPoints, approxPoints);
            Mat M = getPerspectiveTransform(orderedPoints, destPoints);
            Mat warped;
            warpPerspective(img, warped, M, cv::Size(width, height));

            if (warped.cols > warped.rows) // came out landscape
                cv::rotate(warped, warped, cv::ROTATE_90_CLOCKWISE);
            
            
            imshow("lines", lines);
            moveWindow("lines", 0, 0);      // offset by ~width of first window
            imshow("warped", warped);
            moveWindow("warped", 700, 0);      // offset by ~width of first window
            waitKey(0);
            destroyAllWindows();
        }
    }
}

Mat getCardFromImage(Mat& img)
{
    // Loads an image and greyscale it.
    Mat src;
    cvtColor(img, src, COLOR_BGR2GRAY);

    // Get threshold of image.
    threshold(src, src, 196, 255, 0);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    //Finds all contours
    findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& cnt : contours)
    {
        // arbitrary size  for now, works well and deletes tiny specs
        if (contourArea(cnt) > 6000)
        {   
            // Idenfity conotur rectangle and get points of edges
            std::vector<cv::Point> approxPoints;
            auto perimeter = arcLength(cnt, true);
            approxPolyDP(cnt, approxPoints, 0.02 * perimeter, true);

            // If shape isnt a rectangle with 4 points or concave we can skip.
            if (approxPoints.size() != 4 || (!isContourConvex(approxPoints))) continue;

            for (int i = 0; i < 4; i++)
            {
                line(img, approxPoints[i], approxPoints[(i + 1) % 4], Scalar(0, 255, 0), 7);
            }

            // dimensiones of warped image. 6.1cm/8.6cm asepct ratio
            const int width = 610;
            const int height = 860;
            std::vector<cv::Point2f> destPoints = {{0, 0}, {width, 0}, {width, height}, {0, height}};
            
            // find angle of rotation
            // Convert to how far off from nearest 90 e.g.
            auto rect = minAreaRect(approxPoints);

            std::vector<Point2f> rectPoints;
            rect.points(rectPoints);

            for (int i = 0; i < 4; i++)
            {
                line(img, rectPoints[i], rectPoints[(i + 1) % 4], Scalar(255,0,0), 7);
            }

            float angle = rect.angle;
            float snapAngle = angle;
            if (snapAngle < -45.0f) { snapAngle += 90.0f; }

            std::cout << "rect.angle: " << rect.angle 
          << " snapAngle: " << snapAngle << "\n";
            
            // snapAngle is now always in (-45, 45] — the minimum correction to snap to 90
            auto rotatedPoints = rotateContour(approxPoints, -snapAngle);
            

            for (int i = 0; i < 4; i++)
            {
                line(img, rotatedPoints[i], rotatedPoints[(i + 1) % 4], Scalar(0,0,255), 7);
            }
            
            auto orderedPoints = orderPoints(rotatedPoints, approxPoints);
            Mat M = getPerspectiveTransform(orderedPoints, destPoints);
            Mat warped;
            warpPerspective(img, warped, M, cv::Size(width, height));

            return warped;
        }
    }
    return Mat(); // no card found
}


int main()
{

    // for (int i = 0; i < 7; i++)
    // {
    //     std::cout << i << "\n"; 
    //     const String filename = "images/card" +  to_string(i + 1) + ".jpeg";

    //     getCardFromImage(filename);
    // }
    
    
    // return 0;

    cv::VideoCapture cap(0); // 0 = default camera
    
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

        Mat warped = getCardFromImage(frame);

        if (!warped.empty())
        {
            // resize warped to match frame height
            Mat warpedResized;
            cv::resize(warped, warpedResized, cv::Size(frame.rows * (6.1 / 8.6), frame.rows));

            // stitch side by side
            Mat composite;
            cv::hconcat(frame, warpedResized, composite);

            imshow("card scanner", composite);
        }
        else
        {
            imshow("card scanner", frame); // no card found, just show frame
        }

        if (waitKey(1) == 'q') break;
    }
}

/*

                Mat lines = img.clone();
            // for (int i = 0; i < 4; i++)
            // {
            //     line(lines, approxPoints[i], approxPoints[(i + 1) % 4], Scalar(0,0,255), 7);
            // }

            // imshow("lines", lines);
            // waitKey(0);
            // destroyAllWindows();


    if (contourArea(cnt) > 5000)
        {
            std::vector<cv::Point> approx;
            
            double peri = arcLength(cnt, true); //perimiter length.

            approxPolyDP(cnt, approx, 0.02 * peri, true); // approximation of the corners

            if (approx.size() == 4 && isContourConvex(approx))
            {
                for (int i = 0; i < 4; i++)
                {
                    line(img, approx[i], approx[(i + 1) % 4], Scalar(0, 0, 255), 5);
                }
                
                auto ordered = orderPoints(approx);
                
                int cardW = 400, cardH = 600; // desired output size
                std::vector<cv::Point2f> dst = {
                    {0, 0},
                    {(float)cardW, 0},
                    {(float)cardW, (float)cardH},
                    {0, (float)cardH}
                };
                
                cv::Mat M = getPerspectiveTransform(ordered, dst);
                cv::Mat warped;

                warpPerspective(src, warped, M, cv::Size(cardW, cardH));
                

                imshow("img2", img);
                imshow("card", warped);
                waitKey(0);
            }
        }
            
        */