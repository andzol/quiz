#pragma once
#include <opencv2/opencv.hpp>

#include <iterator>
#include <vector>
#include <cmath>
#include <math.h>

#define EPS 0.00000001

using namespace cv;
using namespace std;

int haralik_main(int argc, char **argv);
bool OutofBounds(int i, int j, Mat img);
float Entropy(float* &vec, int len);
void meanStd(float* &vec,  int len, float &m, float &stdev);
Mat get_global_features(Mat img);

class HaralickExtractor {
private:
	Mat matcooc; //GLCM
	float* margprobx;
	float* margproby;
	int probsum_size;
	float* probsum; //sum probability
	int probdiff_size;
	float* probdiff; //diff probability
	float hx, hy; //entropy of margprobx and y
	float meanx, meany, stddevx, stddevy;
	bool initial; //marks if above variables are set

						  /*calculates probsum, probdiff, margprobx and y at once*/
	void fast_init() {
		if (matcooc.empty())
			return;
		/*margprobx.clear();
		margprobx.resize(matcooc.rows, 0.0);
		margproby.clear();
		margproby.resize(matcooc.cols, 0.0);
		probsum.clear();
		probsum.resize(matcooc.rows * 2, 0.0);
		probdiff.clear();
		probdiff.resize(matcooc.rows, 0.0);*/
		int m1 = matcooc.rows * sizeof(float);
		int m2 = matcooc.cols * sizeof(float);
		probdiff = (float*)malloc(m1);
		probdiff_size = matcooc.rows;
		memset(probdiff, 0, m1);
		margprobx = (float*) malloc(m1);
		memset(margprobx, 0, m1);
		margproby = (float*)malloc(m2);
		memset(margproby, 0, m2);
		probsum = (float*)malloc(m1 * 2);
		probsum_size = matcooc.rows;
		memset(probsum, 0, m1 * 2);
		double local;
		for (int i = 0; i < matcooc.rows; i++) {
			for (int j = 0; j < matcooc.cols; j++) {
				local = matcooc.at<float>(i, j);
				margprobx[i] += local;
				margproby[j] += local;
				probsum[i + j] += local;
				probdiff[abs(i - j)] += local;
			}
		}
		hx = Entropy(margprobx, matcooc.rows);
		hy = Entropy(margproby, matcooc.cols);
		meanStd(margprobx, matcooc.rows, meanx, stddevx);
		meanStd(margproby, matcooc.cols, meany, stddevy);
		//Everything set up
		initial = true;
	}

	/*0 => energy, 1 => entropy, 2=> inverse difference */
	/*3 => correlation, 4=> info measure 1, 5 => info measure 2*/
	void cooc_feats(float* ans) {
		float hxy1 = 0.0;
		float hxy2 = 0.0;
		float local;
		//_matcooc=malloc(sizeof(double)*matcooc.rows)
		for (int i = 0; i < matcooc.rows; i++) {
			for (int j = 0; j < matcooc.cols; j++) {
				local = matcooc.at<float>(i, j);
				ans[0] += local * local;
				ans[1] += local * log(local + EPS);
				ans[2] += local * (1 / (1 + (i - j) * (i - j)));
				ans[3] += (i * j * local) - (meanx * meany);
				hxy1 += local * log(margprobx[i] * margproby[j] + EPS);
				hxy2 += margprobx[i] * margproby[j] * log(margprobx[i] * margproby[j] + EPS);
			}
		}
		hxy1 = hxy1 * -1;
		hxy2 = hxy2 * -1;
		ans[1] = -1 * ans[1];
		ans[3] = ans[3] / (stddevx * stddevy);
		ans[4] = (ans[1] - hxy1) / max(hx, hy);
		ans[5] = sqrt(1 - exp(-2 * (hxy2 - ans[1])));
	}

	/*0 => contrast, 1 => diff entropy, 2 => diffvariance */
	/*3 => sum average, 4 => sum entropy, 5 => sum variance */
	void margprobs_feats(float* ans) {
		for (int i = 0; i < probdiff_size; i++) {
			ans[0] += i * i * probdiff[i];
			ans[1] += -1 * probdiff[i] * log(probdiff[i] + EPS);
		}
		for (int i = 0; i < probsum_size; i++) {
			ans[3] += i * probsum[i];
			ans[4] += -1 * probsum[i] * log(probsum[i] + EPS);
		}
		for (int i = 0; i < probdiff_size; i++)
			ans[2] += (i - ans[1]) * (i - ans[1]) * probdiff[i];
		for (int i = 0; i < probsum_size; i++)
			ans[5] += (i - ans[4]) * (i - ans[4]) * probsum[i];
	}



public:
	void fast_feats(float *result) {
		//vector<double> result(12, 0.0);
		if (matcooc.empty()) {
			return;
		}
		if (!initial)
			fast_init();
		float margfeats[6];
		margprobs_feats(margfeats);
		float coocfeats[6];
		cooc_feats(coocfeats);
		for (int i = 0; i < 6; i++)
			result[i] = coocfeats[i];
		for (int i = 0; i < 6; i++)
			result[6 + i] = margfeats[i];
	}

	Mat MatCooc(Mat img, int N, int deltax, int deltay) {
		int target, next;
		int newi, newj;
		Mat ans = Mat::zeros(N + 1, N + 1, CV_32F);
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				newi = i + deltay;
				newj = j + deltax;
				if (newi < img.rows && newj < img.cols && newj >= 0 && newi >= 0) {
					target = (int)img.at<uchar>(i, j);
					next = (int)img.at<uchar>(newi, newj);
					ans.at<float>(target, next) += 1.0;
				}
			}
		}
		return ans / (img.rows * img.cols);
	}

	Mat MatCoocAdd(Mat img, int N, vector<int> deltax, vector<int> deltay) {
		Mat ans, nextans;
		ans = MatCooc(img, N, deltax[0], deltay[0]);
		for (size_t i = 1; i < deltax.size(); i++) {
			nextans = MatCooc(img, N, deltax[i], deltay[i]);
			add(ans, nextans, ans);
		}
		return ans;
	}

	void getFeaturesFromImage(float* result,Mat img, vector<int> deltax, vector<int> deltay) {
		if (img.type() != CV_8UC1) {
			cout << "Unsupported image type" << endl;
			return;
		}
		matcooc = MatCoocAdd(img, 255, deltax, deltay);
		fast_init(); //initialize internal variables
		fast_feats(result);
		/*if (normalize) {
			cv::normalize(ans, ans, 1.0, 0.0,norm_type );
		}*/
	}

	//Constructor for use on single image
	//img is a grayscale image, deltax and deltay are pairs of the directions
	//to which we want to make the GLCM
	//temporarily accepting only CV_8UC1
	//HaralickExtractor(Mat img, vector<int> deltax, vector<int> deltay) {
	//	if (img.type() != CV_8UC1) {
	//		cout << "Unsupported image type" << endl;
	//		return;
	//	}
	//	matcooc = MatCoocAdd(img, 255, deltax, deltay);
	//}

	//Constructor for use on various images
	HaralickExtractor():
		margprobx(NULL),
		margproby(NULL),
		probsum(NULL),
		probdiff(NULL),initial(false) {
		return;
	}
	~HaralickExtractor(){
		if (margprobx != NULL) {
			free(margprobx);
		}
		if (margproby != NULL) {
			free(margproby);
		}
		if (probsum != NULL) {
			free(probsum);
		}
		if (probdiff != NULL) {
			free(probdiff);
		}
	}
};
