#include <opencv2/opencv.hpp>       // opencv general include file
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "order.h"
#define ALOGE(...)
#define ALOGV(...)
//#include "feat.h"


using namespace cv;
using namespace std;
typedef enum attr_t {
	black,
	blue,
	red,
	green
} atype;
typedef struct mi_t {
	atype type;
	int x;
	int y;

} mi;
#define RGBA_8888_BPP 4
#define RGB_565_BPP 2
typedef unsigned char u8;  // in case char is signed by default on your platform
#ifndef uint16_t
typedef unsigned short uint16_t;//
#endif
void test_arr(mi** arr, int* arr_len) {
	int len = 13;
	mi* results =(mi*) malloc(len * sizeof(mi));
	for (int i = 0; i < len; i++) {
		results[i].x = i+1;
		results[i].y = i;
		results[i].type = static_cast<atype>(i % 4);
	}
	*arr = results;
	*arr_len = len;
	(*arr_len)++;

}
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
	int result,bpp=-1;
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

int main(int argc, char** argv)
{
	cv::Mat mat,sample;
	char rgba8888[512],rgb565[512],sampleFN[512];
	sprintf_s(rgba8888, "%s\\ankulua.raw", argv[1]);
	sprintf_s(rgb565, "%s\\sc.raw", argv[1]);
	sprintf_s(sampleFN, "%s\\sample.png", argv[1]);
	mat=readRaw(rgb565);
	imshow("rgb565",mat);
	mat=readRaw(rgba8888);
	//imshow("rgba8888", mat);
	sample = imread(sampleFN, IMREAD_COLOR);
	cv::cvtColor(sample, sample, cv::COLOR_BGRA2BGR);
	//assert(mat.type() == sample.type());
	int len;
	mi *results=NULL;
	test_arr(&results, &len);
	//if (argc != 2)
	//{
	//	cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
	//	return -1;
	//}

	//Mat image;
	//image = imread(argv[1], IMREAD_COLOR); // Read the file

	//if (!image.data) // Check for invalid input
	//{
	//	cout << "Could not open or find the image: " << std::string(argv[1]) <<std::endl;
	//	return -1;
	//}
	////haralik_main(argc, argv);
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", mat); // Show our image inside it.

	
	for (int i = 0; i < len; i++) {
		printf("x:%d, y:%d, type: %d\n", results[i].x, results[i].y, results[i].type);
	}
	free(results);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}