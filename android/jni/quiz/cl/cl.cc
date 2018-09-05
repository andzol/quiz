#include <iostream>
#include <error.h>
#include <string>
#include <vector>
#include "qlog.h"
#include "feat.h"
#include <opencv2/opencv.hpp>       // opencv general include file
#include "load_classifier.h"
#include "cl.h"
//#include <ml.h>		  // opencv machine learning include file


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <math.h>


using namespace cv;
using namespace std;

#define RGBA_8888_BPP 4
#define RGB_565_BPP 2
typedef unsigned char u8;  // in case char is signed by default on your platform
#ifndef uint16_t
typedef unsigned short uint16_t;//
#endif


int thresh = 50, N = 5;
static CvRTrees classifier;


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
				
				if (approx.size() == 4 &&
					fabs(contourArea(Mat(approx))) > 7000 &&
					fabs(contourArea(Mat(approx))) < 50000 &&
					isContourConvex(Mat(approx))
					)
				{
					double lenX = norm(approx[1] - approx[0]);
					double lenY = norm(approx[2] - approx[1]);
					double aspectRatio = (lenX < lenY ? lenX / lenY : lenY / lenX);
					if (aspectRatio < 0.95) {
						continue;//not a square.
					}
					//cout << "ratio << "<< aspectRatio << endl;
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
					if (maxCosine < 0.1) {
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

/*
// the function draws all the squares in the image
static void drawSquares(Mat& image, const vector<vector<Point> >& squares, CvRTrees &classifier)
{
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];
		char imgName[32];
		int n = (int)squares[i].size();
		//dont detect the border
		if (p->x > 3 && p->y > 3) {
			
			Point massCenter = findCountourCenter(squares[i]);
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			circle(image, massCenter, 9, color, -1, 8, 0);
			ostringstream centerXY;
			centerXY << "(" << massCenter.x << "," << massCenter.y << ")";
			putText(image, centerXY.str(), massCenter+Point(0,20), FONT_HERSHEY_SIMPLEX, 1, color, 3);
			//polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, CV_AA);
			Mat mask = Mat::zeros(image.size(), CV_8UC1);
			vector<Point> roi(p, p+ n);
			fillConvexPoly(mask, roi, Scalar(255, 255, 255));
			Mat img;
			bitwise_and(image, image, img,mask);
			Rect rect = boundingRect(roi);
			image(rect).copyTo(img);
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
*/


static inline void rgb565pixrgb(uint16_t rgb565, u8 &r, u8 &g,u8 &b)
{
	r = ((((rgb565 >> 11) & 0x1F) * 527) + 23) >> 6;
	g = ((((rgb565 >> 5) & 0x3F) * 259) + 33) >> 6;
	b = (((rgb565 & 0x1F) * 527) + 23) >> 6;
}
static inline cv::Mat readRGB565(const int height, const int width, FILE* f, void* pixData) {
	u8 r, g, b;
	u8* pd = NULL;
	uint16_t chunk;
	Vec3b bgr;
	Mat img = Mat::zeros(height, width, CV_8UC3);
	pixData = (u8*)malloc(width*height * RGB_565_BPP * sizeof(u8));
	// 2bpp. rgba_565 format
	ALOGV("assume we have raw RGB_565 image[%dx%d]", width, height);
	fread(pixData, width*height * RGB_565_BPP, 1, f);
	assert(feof(f) == 0);//at eof?
	pd = (u8*) pixData;
	for (unsigned _r = 0; _r < height; ++_r) {
		for (unsigned _c = 0; _c < width; ++_c) {
			chunk = *(pd++);
			chunk |= *(pd++) << 8;
			rgb565pixrgb(chunk, r, g, b);
			bgr[0] = b; bgr[1] = g; bgr[2] = r;
			img.at<cv::Vec3b>(_r, _c) = bgr;
		}
	}
	return img;
}
static inline cv::Mat readRGBA888(const int height,const int width,FILE* f,void* pixData) {
	u8 r, g, b, a;
	u8* pd = NULL;
	Mat img = Mat::zeros(height, width, CV_8UC4);
	pixData = (u8*)malloc(width*height * RGBA_8888_BPP * sizeof(u8));
	// 4 bpp. rgba_888 format
	ALOGV("assume we have raw RGBA_8888 image[%dx%d]", width, height);
	fread(pixData, width*height * RGBA_8888_BPP, 1, f);
	assert(feof(f)==0);//at eof?
	pd =(u8*) pixData;
	for (unsigned _r = 0; _r < height; ++_r) {
		for (unsigned _c = 0; _c < width; ++_c) {
			Vec4b bgra;
			r = *(pd++)/*red*/;
			g = *(pd++)/*green*/;
			b = *(pd++)/*blue*/;
			a = *(pd++) /*alpha*/;
			//Image in OpenCV are in BGRA, not in RGBA!
			bgra[2] = r;
			bgra[1] = g;
			bgra[0] = b;
			bgra[3] = a;
			img.at<cv::Vec4b>(_r, _c) = bgra;
			//i = pixel;
		}
	}

	ALOGV("bgra->bgr convert");
	cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
	return img;
}
/*
from android/include/system/graphics.h
HAL_PIXEL_FORMAT_RGBA_8888          = 1,
HAL_PIXEL_FORMAT_RGBX_8888          = 2,
HAL_PIXEL_FORMAT_RGB_888            = 3,
HAL_PIXEL_FORMAT_RGB_565            = 4,
HAL_PIXEL_FORMAT_BGRA_8888          = 5,
HAL_PIXEL_FORMAT_RGBA_5551          = 6,
HAL_PIXEL_FORMAT_RGBA_4444          = 7,

from .../include/ui/PixelFormat.h
// real pixel formats supported for rendering -----------------------------
PIXEL_FORMAT_RGBA_8888   = HAL_PIXEL_FORMAT_RGBA_8888,  // 4x8-bit RGBA
PIXEL_FORMAT_RGBX_8888   = HAL_PIXEL_FORMAT_RGBX_8888,  // 4x8-bit RGB0
PIXEL_FORMAT_RGB_888     = HAL_PIXEL_FORMAT_RGB_888,    // 3x8-bit RGB
PIXEL_FORMAT_RGB_565     = HAL_PIXEL_FORMAT_RGB_565,    // 16-bit RGB
PIXEL_FORMAT_BGRA_8888   = HAL_PIXEL_FORMAT_BGRA_8888,  // 4x8-bit BGRA
PIXEL_FORMAT_RGBA_5551   = HAL_PIXEL_FORMAT_RGBA_5551,  // 16-bit ARGB
PIXEL_FORMAT_RGBA_4444   = HAL_PIXEL_FORMAT_RGBA_4444,  // 16-bit ARGB
PIXEL_FORMAT_A_8         = GGL_PIXEL_FORMAT_A_8,        // 8-bit A
PIXEL_FORMAT_L_8         = GGL_PIXEL_FORMAT_L_8,        // 8-bit L (R=G=B=L)
PIXEL_FORMAT_LA_88       = GGL_PIXEL_FORMAT_LA_88,      // 16-bit LA
PIXEL_FORMAT_RGB_332     = GGL_PIXEL_FORMAT_RGB_332,    // 8-bit RGB

from .../cmds/screencap/screencap.cpp
	switch (vinfo.bits_per_pixel) {
	case 16:
	*f = PIXEL_FORMAT_RGB_565;
	*bytespp = 2;
	break;
	case 24:
	*f = PIXEL_FORMAT_RGB_888;
	*bytespp = 3;
	break;
	case 32:
	// TODO: do better decoding of vinfo here
	*f = PIXEL_FORMAT_RGBX_8888;
	*bytespp = 4;
	break;
	default:
	return BAD_VALUE;
	}
*/
static cv::Mat readRaw(const char* fn) {
	unsigned width, height, pixelFormat;
	int result;
	Mat img;
	FILE * f;
	void* pixData = NULL;
	u8 header[12];
	//u8 buf[1024 * 1024];
	f = fopen(fn, "rb");
	if (!f) {
		ALOGE("can't read from %s. errno: %d, error: %s", fn, errno, strerror(errno));
		goto finish;
	}
	result = fread(header, 1, 12, f);
	if (result != 12) {
		ALOGE("can't read header from %s", fn);
		goto finish;
	}
	width = ((u8)header[3] << 24) | ((u8)header[2] << 16) | ((u8)header[1] << 8) | (u8)header[0];
	height = ((u8)header[7] << 24) | ((u8)header[6] << 16) | ((u8)header[5] << 8) | (u8)header[4];
	pixelFormat = ((u8)header[11] << 24) | ((u8)header[10] << 16) | ((u8)header[9] << 8) | (u8)header[8];
	switch (pixelFormat) {
	case 1://RGBA_8888
		img = readRGBA888(height, width, f, pixData);
		break;
	case 4://RGB_565
		img = readRGB565(height, width, f, pixData);
		break;
	default:
		ALOGE("%s has wrong/unsupported pixel format %d", fn, pixelFormat);
	}
finish:
	if (pixData != NULL) {
		free(pixData);
	}
	if (f) {
		fclose(f);
	}
	return img;// returns BGR COLOR image
}




int cl_init() {
	if (ReadForest(classifier)) {
		ALOGV("classifier initialized");
		return 1;
	}
	return 0;
}

MonstersInfo* cl_recognize(const char* filename) {
//,MonsterInfo **results,int *results_len) {
	int results_len=0;
	MonsterInfo* mi=NULL;
	MonstersInfo* msi=NULL;
	vector<vector<Point> > squares;
	ALOGV("readRaw from %s",filename);
	cv::Mat image=readRaw(filename);
	if (image.empty()) {
		ALOGV("empty image");
		return NULL;
	}
	ALOGV("findSquares");
	findSquares(image,squares);
	int roiSize = 0;
	ALOGV("detect roi");
	for(size_t i=0;i<squares.size();i++) {
		const Point* p=&squares[i][0];
		if (p->x >3 && p->y >3) {
		  roiSize++;
		}
	}
	mi = (MonsterInfo*) malloc(roiSize*sizeof(MonsterInfo));
	results_len=roiSize;
	ALOGV("found %d squares on image. roi count: %d",squares.size(),roiSize);
        for (size_t i = 0; i < squares.size(); i++)
        {
                const Point* p = &squares[i][0];
                int n = (int)squares[i].size();
                //dont detect the border
                if (p->x > 3 && p->y > 3) {
			roiSize=roiSize-1;
                        Point massCenter = findCountourCenter(squares[i]);
                        Mat mask = Mat::zeros(image.size(), CV_8UC1);
                        vector<Point> roi(p, p+ n);
                        fillConvexPoly(mask, roi, Scalar(255, 255, 255));
                        Mat img;
                        bitwise_and(image, image, img,mask);
                        Rect rect = boundingRect(roi);
                        image(rect).copyTo(img);
                        Mat gf = get_global_features(img);
                        float pred = classifier.predict(gf, Mat());
			mi[roiSize].x=massCenter.x;
			mi[roiSize].y=massCenter.y;
			mi[roiSize].attr=(MonsterType)pred;
			
                }

        }
	assert(0 == roiSize);//all processed
	msi=(MonstersInfo*) malloc(sizeof(MonstersInfo));
	msi->mi=mi;
	msi->mi_size=results_len;
	ALOGV("recognition done. got %d results",results_len);
	return msi;
}
/*
int main(int argc, char** argv)
{
	namedWindow(wndname, 1);
	vector<vector<Point> > squares;
	string quizImage = string(argv[1]) + "\\quiz_img\\img1.jpg";
	string rfcfn = string(argv[1]) + "\\tt_dataset\\pretrained.xml";
	CvRTrees classifier;
	//classifier.load(rfcfn.c_str());
	ReadForest(classifier);
	Mat image = imread(quizImage.c_str(), 1);
	if (image.empty())
	{
		//cout << "Couldn't load " << quizImage << endl;
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

*/
