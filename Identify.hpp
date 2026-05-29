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

enum class SUIT { HEART, DIAMOND, CLUB, SPADE };
enum class RANK { ACE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING };
enum class COLOR { BLACK, RED, UNKNOWN };

using CARD = std::pair<RANK, SUIT>; 

void initTemplates();

SUIT identifySuit(Mat& imgCorner, COLOR suitColor);
String numToSuit(int suit);
CARD identifyCard(Mat& imgCorner, Mat& imgCard);