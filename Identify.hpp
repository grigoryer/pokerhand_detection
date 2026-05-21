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

void initTemplates();

SUIT identitySuit(Mat& imgCard);