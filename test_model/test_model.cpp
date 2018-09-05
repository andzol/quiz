#include <Windows.h>
#include <iostream>
#include <string>
#include <vector>
#include "feat.h"
#include <opencv2/opencv.hpp>       // opencv general include file
#include "load_classifier.h"
#include "test_model.h"
//#include <ml.h>		  // opencv machine learning include file


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <math.h>

#define WATER 1
#define FIRE 2;
#define DARK 3;
#define LIGHT 4;
#define WIND 5;



typedef int(*AssumeFunct)(int,string&);

using namespace cv;
using namespace std;

RNG rng(12345);


static bool hasEnding(std::string const &fullString, std::string const &ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	else {
		return false;
	}
}
static void FindFiles(const std::string &directory, vector<string> &files)
{

	std::string tmp = directory + "\\*";
	WIN32_FIND_DATA file;
	HANDLE search_handle = FindFirstFile(tmp.c_str(), &file);
	if (search_handle != INVALID_HANDLE_VALUE)
	{
		std::vector<std::string> directories;

		do
		{

			tmp = directory + "\\" + std::string(file.cFileName);
			if (file.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				if ((!lstrcmp(file.cFileName, ".")) || (!lstrcmp(file.cFileName, "..")))
					continue;
				directories.push_back(tmp);
				continue;
			}
			if (hasEnding(tmp, string(".png"))) {
				files.push_back(tmp);

			}

		} while (FindNextFile(search_handle, &file));

		FindClose(search_handle);

		for (std::vector<std::string>::iterator iter = directories.begin(), end = directories.end(); iter != end; ++iter)
			FindFiles(*iter, files);
	}
}
static inline string getSampleType(float &p) {
	if (p==2) return "fire";
	if (p==5) return "wind";
	if (p==1) return "water";
	if (p==3) return "dark";
	if (p==4) return "light";
	return "xxx";
}
static inline uchar getSampleType(string &fn) {
	if (string::npos != fn.find("fire")) return FIRE;
	if (string::npos != fn.find("wind")) return WIND;
	if (string::npos != fn.find("water")) return WATER;
	if (string::npos != fn.find("dark")) return DARK;
	if (string::npos != fn.find("light")) return LIGHT;
	return 0;
}
static int assumedType(string fn) {
	if (string::npos != fn.find("test\\f")) return FIRE;
	if (string::npos != fn.find("test\\w")) return WIND;
	if (string::npos != fn.find("test\\a")) return WATER;
	if (string::npos != fn.find("test\\d")) return DARK;
	if (string::npos != fn.find("test\\l")) return LIGHT;
	return -1;
}

static void test(CvRTrees &classifier, vector<string> &test_files, AssumeFunct f) {
	
	vector<int> results;
	for (vector<string>::iterator iter = test_files.begin(), end = test_files.end(); iter != end; ++iter) {
		Mat m = imread(iter->c_str(), IMREAD_COLOR);
		Mat gf = get_global_features(m);
		float pred = classifier.predict(gf, Mat());
		//cout << *iter << " prediction: " << pred << endl;
		int correct = f(pred,*iter);// 
		results.push_back(correct);
	}
	Mat r = Mat(results);
	Scalar r_sum = cv::sum(r);
	double v = r_sum.val[0];
	cout << "testing result: " << v << " of " << results.size() << " images. error rate is: " << 100 * (results.size() - v) / results.size() << "%" << endl;
}

int unseen_assume(int pred,string& str) {
	return pred == assumedType(str) ? 1 : 0;
}

int seen_assume(int p, string& str) {
	return p == getSampleType(str) ? 1 : 0;
}

int test_main(int argc, char **argv) {

	if (argc != 2)
	{
		cout << " Usage: display_image ...." << endl;
		return -1;
	}
	string rfcfn = string(argv[1]) + "\\pretrained.xml";
	CvRTrees classifier;
	classifier.load(rfcfn.c_str());
	vector<string> test_files;
	FindFiles(string(argv[1]) + "\\test", test_files);
	cout << "test over unseen images" << endl;
	test(classifier, test_files, &unseen_assume);
	cout << "test over seen images" << endl;
	test_files.clear();
	FindFiles(string(argv[1]) + "\\train", test_files);
	test(classifier, test_files,&seen_assume);
	//waitKey(0); // Wait for a keystroke in the window

}



int thresh = 50, N = 5;
const char* wndname = "icons find and type detect";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
}

static Point findCountourCenter(cv::InputArray &inp) {
	Moments mu = cv::moments(inp);
	Point massCenter = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
	return massCenter;
}
//slow :(
static bool nearbyCenterExists(Point& mc, vector<Point> inputs, double delta=5/*px?*/) {
	for (vector<Point>::iterator iter = inputs.begin(), end = inputs.end(); iter != end; ++iter) {
		Point poi = *iter;
		double euqDist = cv::norm(mc - poi);
		if (euqDist< delta) {
			//cout <<"nearby point: "<< mc << "euq dist: " << euqDist << endl;
			return true;
		}
	}
	return false;
}
// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
	squares.clear();

	//s    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	// down-scale and upscale the image to filter out the noise
	//pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
	//pyrUp(pyr, timg, image.size());


	// blur will enhance edge detection
	Mat timg;
	medianBlur(image, timg, 9);
	imshow("medianBlur", timg);
	Mat gray0(timg.size(), CV_8U), gray;

	vector<vector<Point> > contours;
	vector<Point> contourCenters;
	int numChannels = image.channels(); //3
	// find squares in every color plane of the image
	for (int c = 0; c < numChannels; c++)
	{
		int ch[] = { c, 0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		// try several threshold levels
		for (int l = 0; l < N; l++)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			if (l == 0)
			{
				// apply Canny. Take the upper threshold from slider
				// and set the lower to 0 (which forces edges merging)
				Canny(gray0, gray, 5, thresh, 5);
				// dilate canny output to remove potential
				// holes between edge segments
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				// apply threshold if l!=0:
				//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}

			// find contours and store them all as a list
			findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			vector<Point> approx;

			// test each contour
			for (size_t i = 0; i < contours.size(); i++)
			{
				
				// approximate contour with accuracy proportional
				// to the contour perimeter
				
				approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

				// square contours should have 4 vertices after approximation
				// relatively large area (to filter out noisy contours)
				// and be convex.
				// Note: absolute value of an area is used because
				// area may be positive or negative - in accordance with the
				// contour orientation
				double _carea = fabs(contourArea(Mat(approx)));
				if (approx.size() == 4 &&
					_carea >3000 &&
					_carea <50000 &&
					//fabs(contourArea(Mat(approx))) > 7000 &&
					//fabs(contourArea(Mat(approx))) < 50000 &&
					isContourConvex(Mat(approx))
					)
				{
					double lenX = norm(approx[1] - approx[0]);
					double lenY = norm(approx[2] - approx[1]);
					double aspectRatio = (lenX < lenY ? lenX / lenY : lenY / lenX);
					if (aspectRatio < 0.95) {
						continue;//not a square.
					}
					cout << "area: "<< _carea << "; ratio << "<< aspectRatio << endl;
					double maxCosine = 0;

					for (int j = 2; j < 5; j++)
					{
						// find the maximum cosine of the angle between joint edges
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
					}

					// if cosines of all angles are small
					// (all angles are ~90 degree) then write quandrange
					// vertices to resultant sequence
					if (maxCosine < 0.2) {
						Point massCenter = findCountourCenter(contours[i]);
						// check if we have no same points nearby. this will prevent almost similar contours, found on different channels
						if (!nearbyCenterExists(massCenter, contourCenters)) {
							squares.push_back(approx);
							contourCenters.push_back(massCenter);
						}
						
						
					}
						
				}
			}
		}
	}
}


// the function draws all the squares in the image
static void drawSquares(Mat& image, const vector<vector<Point> >& squares, CvRTrees &classifier)
{
	Mat original;
	image.copyTo(original);
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];
		char imgName[32];
		int n = (int)squares[i].size();
		//dont detect the border
		if (p->x > 3 && p->y > 3) {
			string name = ("mobicon" + i);
			Point massCenter = findCountourCenter(squares[i]);
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			circle(image, massCenter, 9, color, -1, 8, 0);
			ostringstream centerXY;
			centerXY << "(" << massCenter.x << "," << massCenter.y << ")";
			putText(image, centerXY.str(), massCenter+Point(0,20), FONT_HERSHEY_SIMPLEX, 1, color, 3);
			polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, CV_AA);
			Mat mask = Mat::zeros(image.size(), CV_8UC1);
			vector<Point> roi(p, p+ n);
			fillConvexPoly(mask, roi, Scalar(255, 255, 255));
			Mat img;
			//bitwise_and(original, original, img,mask);
			Rect rect = boundingRect(roi);
			original(rect).copyTo(img);
			imshow(name.c_str(), img);
			Mat gf = get_global_features(img);
			float pred = classifier.predict(gf, Mat());
			string predictedType= getSampleType(pred);
			Scalar tColor = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			putText(image, predictedType, massCenter- Point(0, 20), FONT_HERSHEY_SIMPLEX,1,tColor,3);
			//imshow(ss.str().c_str(), img);
		}
			
	}
	resize(image, image, Size(1280, 720));
	imshow(wndname, image);
}


int main(int argc, char** argv)
{
	namedWindow(wndname, 1);
	vector<vector<Point> > squares;
	//string quizImage = string(argv[1]) + "\\quiz_img\\img1.jpg";
	string quizImage = string(argv[1]) + "\\quiz_img\\Screenshot_2018-08-07-05-40-34.png";
	string rfcfn = string(argv[1]) + "\\tt_dataset\\pretrained.xml";
	CvRTrees classifier;
	//classifier.load(rfcfn.c_str());
	ReadForest(classifier);
	Mat image = imread(quizImage.c_str(), 1);
	if (image.empty())
	{
		cout << "Couldn't load " << quizImage << endl;
		return -1;
	}
	findSquares(image, squares);
	drawSquares(image, squares, classifier);
	while (true) {
		int c = waitKey();
		if ((char)c == 27)
			break;
	}


	return 0;
}