
#include "feat.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
using namespace std;

static std::vector<int> deltax({ 1 });
static std::vector<int> deltay({ 0 });

bool OutofBounds(int i, int j, Mat img) {
	return (i > img.rows || i < 0 && j > img.cols && j < 0);
}

float Entropy(float* &vec,int len) {
	float result = 0.0;
	for (int i = 0; i < len; i++)
		result += vec[i] * log(vec[i] + EPS);
	return -1 * result;
}

void meanStd(float* &v, int len, float &m, float &stdev) {
	float sum = 0.0;
	for(int i=0;i!=len;i++) {
		sum += v[i];
	}
	m = sum / len;

	float accum = 0.0;
	for (int i = 0; i != len; i++) {
		accum += (v[i] - m) * (v[i] - m);
	}

	stdev = sqrt(accum / (len - 1));
}

//Marginal probabilities as in px = sum on j(p(i, j))
//                             py = sum on i(p(i, j))
vector<float> MargProbx(Mat cooc) {
	vector<float> result(cooc.rows, 0.0);
	for (int i = 0; i < cooc.rows; i++)
		for (int j = 0; j < cooc.cols; j++)
			result[i] += cooc.at<float>(i, j);
	return result;
}

vector<float> MargProby(Mat cooc) {
	vector<float> result(cooc.cols, 0.0);
	for (int j = 0; j < cooc.cols; j++)
		for (int i = 0; i < cooc.rows; i++)
			result[j] += cooc.at<float>(i, j);
	return result;
}

//probsum  := Px+y(k) = sum(p(i,j)) given that i + j = k
vector<float> ProbSum(Mat cooc) {
	vector<float> result(cooc.rows * 2, 0.0);
	for (int i = 0; i < cooc.rows; i++)
		for (int j = 0; j < cooc.cols; j++)
			result[i + j] += cooc.at<float>(i, j);
	return result;
}

//probdiff := Px-y(k) = sum(p(i,j)) given that |i - j| = k
vector<float> ProbDiff(Mat cooc) {
	vector<float> result(cooc.rows, 0.0);
	for (int i = 0; i < cooc.rows; i++)
		for (int j = 0; j < cooc.cols; j++)
			result[abs(i - j)] += cooc.at<float>(i, j);
	return result;
}


/*Features from coocurrence matrix*/
float HaralickEnergy(Mat cooc) {
	double energy = 0;
	for (int i = 0; i < cooc.rows; i++) {
		for (int j = 0; j < cooc.cols; j++) {
			energy += cooc.at<float>(i, j) * cooc.at<float>(i, j);
		}
	}
	return energy;
}

float HaralickEntropy(Mat cooc) {
	double entrop = 0.0;
	for (int i = 0; i < cooc.rows; i++)
		for (int j = 0; j < cooc.cols; j++)
			entrop += cooc.at<float>(i, j) * log(cooc.at<float>(i, j) + EPS);
	return -1 * entrop;
}

float HaralickInverseDifference(Mat cooc) {
	double res = 0;
	for (int i = 0; i < cooc.rows; i++)
		for (int j = 0; j < cooc.cols; j++)
			res += cooc.at<float>(i, j) * (1 / (1 + (i - j) * (i - j)));
	return res;
}

///*Features from MargProbs */
//double HaralickCorrelation(Mat cooc, vector<double> probx, vector<double> proby) {
//	double corr=0;
//	double meanx, meany, stddevx, stddevy;
//	meanStd(probx, meanx, stddevx);
//	meanStd(proby, meany, stddevy);
//	for (int i = 0; i < cooc.rows; i++)
//		for (int j = 0; j < cooc.cols; j++)
//			corr += (i * j * cooc.at<double>(i, j)) - meanx * meany;
//	return corr / (stddevx * stddevy);
//}
//
////InfoMeasure1 = HaralickEntropy - HXY1 / max(HX, HY)
////HXY1 = sum(sum(p(i, j) * log(px(i) * py(j))
//double HaralickInfoMeasure1(Mat cooc, double ent, vector<double> probx, vector<double> proby) {
//	double hx = Entropy(probx);
//	double hy = Entropy(proby);
//	double hxy1 = 0.0;
//	for (int i = 0; i < cooc.rows; i++)
//		for (int j = 0; j < cooc.cols; j++)
//			hxy1 += cooc.at<double>(i, j) * log(probx[i] * proby[j] + EPS);
//	hxy1 = -1 * hxy1;
//
//	return (ent - hxy1) / max(hx, hy);
//
//}

//InfoMeasure2 = sqrt(1 - exp(-2(HXY2 - HaralickEntropy)))
//HX2 = sum(sum(px(i) * py(j) * log(px(i) * py(j))
float HaralickInfoMeasure2(Mat cooc, float ent, vector<float> probx, vector<float> proby) {
	double hxy2 = 0.0;
	for (int i = 0; i < cooc.rows; i++)
		for (int j = 0; j < cooc.cols; j++)
			hxy2 += probx[i] * proby[j] * log(probx[i] * proby[j] + EPS);
	hxy2 = -1 * hxy2;

	return sqrt(1 - exp(-2 * (hxy2 - ent)));
}

/*Features from ProbDiff*/
float HaralickContrast(Mat cooc, vector<float> diff) {
	float contrast = 0.0;
	for (size_t i = 0; i < diff.size(); i++)
		contrast += i * i * diff[i];
	return contrast;
}

float HaralickDiffEntropy(Mat cooc, vector<float> diff) {
	float diffent = 0.0;
	for (size_t i = 0; i < diff.size(); i++)
		diffent += diff[i] * log(diff[i] + EPS);
	return -1 * diffent;
}

float HaralickDiffVariance(Mat cooc, vector<float> diff) {
	float diffvar = 0.0;
	float diffent = HaralickDiffEntropy(cooc, diff);
	for (size_t i = 0; i < diff.size(); i++)
		diffvar += (i - diffent) * (i - diffent) * diff[i];
	return diffvar;
}

/*Features from Probsum*/
float HaralickSumAverage(Mat cooc, vector<float> sumprob) {
	float sumav = 0.0;
	for (size_t i = 0; i < sumprob.size(); i++)
		sumav += i * sumprob[i];
	return sumav;
}

float HaralickSumEntropy(Mat cooc, vector<float> sumprob) {
	float sument = 0.0;
	for (size_t i = 0; i < sumprob.size(); i++)
		sument += sumprob[i] * log(sumprob[i] + EPS);
	return -1 * sument;
}

float HaralickSumVariance(Mat cooc, vector<float> sumprob) {
	float sumvar = 0.0;
	float sument = HaralickSumEntropy(cooc, sumprob);
	for (size_t i = 0; i < sumprob.size(); i++)
		sumvar += (i - sument) * (i - sument) * sumprob[i];
	return sumvar;
}


Mat MatCooc(Mat img, int N, int deltax, int deltay)
{
	int atual, vizinho;
	int newi, newj;
	Mat ans = Mat::zeros(N + 1, N + 1, CV_64F);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			newi = i + deltay;
			newj = j + deltax;
			if (newi < img.rows && newj < img.cols && newj >= 0 && newi >= 0) {
				atual = (int)img.at<uchar>(i, j);
				vizinho = (int)img.at<uchar>(newi, newj);
				ans.at<float>(atual, vizinho) += 1.0;
			}
		}
	}
	return ans / (img.rows * img.cols);
}

//Assume tamanho deltax == tamanho deltay 
Mat MatCoocAdd(Mat img, int N, std::vector<int> deltax, std::vector<int> deltay)
{
	Mat ans, nextans;
	ans = MatCooc(img, N, deltax[0], deltay[0]);
	for (size_t i = 1; i < deltax.size(); i++) {
		nextans = MatCooc(img, N, deltax[i], deltay[i]);
		add(ans, nextans, ans);
	}
	return ans;
}

void printMat(Mat img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++)
			printf("%lf ", (float)img.at<float>(i, j));
		printf("\n");
	}
}

Mat _huMoments(Mat img) {
	Mat result;
	Moments _moments = cv::moments(img);
	cv::HuMoments(_moments, result);
	//cout << "_huMoments size:" << result.size() << endl;
	result.convertTo(result, CV_32FC1);
	//normalize(result, result, 1.0, 0.0, NORM_MINMAX);
	return result;
}
Mat _hist(Mat img) {
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);
	// Quantize the hue to 30 levels
	// and the saturation to 32 levels
	// and value to 33 levels
	int bins = 16;
	int histSize[] = { bins,bins };
	// hue varies from 0 to 179, see cvtColor
	float hranges[] = { 0, 179 };
	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float sranges[] = { 0, 255 };
	/*Value works in conjunction with saturation and describes the brightness or intensity of the color, 
	from 0 - 100 percent, where 0 is completely black and 100 is the brightest and reveals the most color.*/
	float vranges[] = { 0,100 };
	const float* ranges[] = { hranges, sranges };
	Mat hist;
	// we compute the histogram from the 0-th and 1-st channels
	int channels[] = { 0, 1  };

	calcHist(&hsv, 1, channels, Mat(), // do not use mask
		hist, sizeof(histSize)/sizeof(histSize[0]), histSize, ranges,
		true, // the histogram is uniform
		false);
	//equalizeHist(hist, hist);
	//
	
	//int scale = 10;
	//Mat histImg = Mat::zeros(sbins*scale, hbins * 10, CV_8UC3);

	/*for (int h = 0; h < hbins; h++)
		for (int s = 0; s < sbins; s++)
		{
			float binVal = hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(histImg, Point(h*scale, s*scale),
				Point((h + 1)*scale - 1, (s + 1)*scale - 1),
				Scalar::all(intensity),
				CV_FILLED);
		}

	namedWindow("H-S Histogram", 1);
	imshow("H-S Histogram", histImg);*/
	//Mat hist1d = hist.reshape(1, 1).t();
	//cout << "hist " << hist1d.size() << endl;
	/*Mat _hist = Mat(hist.size(), CV_64F);
	hist.convertTo(_hist, CV_64F);*/

	return hist;
}
//mat is COLOR BGR image
Mat get_global_features(Mat img) {
	resize(img, img, Size(500, 500), 0, 0, INTER_LINEAR);
	Mat img2;
	cvtColor(img, img2, COLOR_BGR2GRAY);
	HaralickExtractor haralik;
	Mat __hist = _hist(img);//this is CV_32FC1
	//convert 3d matrix to 1d array 
	assert(__hist.isContinuous());
	assert(__hist.type() == CV_32FC1);
	const float* p3 = __hist.ptr<float>(0);
	std::vector<float> flat0(p3, p3 + __hist.total());
	
	Mat fd_hist = Mat(flat0);
	cv::normalize(fd_hist, fd_hist, 1.0, 0.0, NORM_MINMAX, CV_32FC1, Mat());
	Mat fd_huMoments = _huMoments(img2);
	float fd_haralik[12];
	haralik.getFeaturesFromImage(fd_haralik,img2, deltax, deltay);
	Mat fd_haralik_m = Mat(12,1, CV_32FC1,fd_haralik,sizeof(float)); //CV_64FC1 is double
	//normalize(fd_haralik_m, fd_haralik_m, 1.0, 0.0, NORM_MINMAX, -1); 
	Mat features;
	vector<Mat> fvec = { fd_hist,fd_huMoments,fd_haralik_m };
	cv::vconcat(fvec, features);
	//Mat norm_features;
	//normalize(features.t(), norm_features, 1.0, 0.0, NORM_MINMAX, CV_32FC1, Mat());
	return features.t();
}
int haralik_main(int argc, char **argv) {
	if (argc < 2) {
		printf("Insira uma imagem\n");
		return 0;
	}

	Mat img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	/*Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat ans = MatCoocAdd(img, 255, deltax, deltay);
	std::vector<double> sum = ProbSum(ans);
	std::vector<double> diff = ProbDiff(ans);
	std::vector<double> probx = MargProbx(ans);
	std::vector<double> proby = MargProby(ans);
	double ent = HaralickEntropy(ans);
	double invdiff = HaralickInverseDifference(ans);
	cout << "Energy: " << HaralickEnergy(ans) << endl;
	cout << "Entropy: " << ent << endl;
	cout << "Inverse Difference: " << invdiff << endl;
	cout << "Correlation: " << HaralickCorrelation(ans, probx, proby) << endl;
	cout << "Info Measure of Correlation 1: " << HaralickInfoMeasure1(ans, ent, probx, proby) << endl;
	cout << "Info Measure of Correlation 2: " << HaralickInfoMeasure2(ans, ent, probx, proby) << endl;
	cout << "Contrast: " << HaralickContrast(ans, diff) << endl;
	cout << "Difference Entropy: " << HaralickDiffEntropy(ans, diff) << endl;
	cout << "Difference Variance: " << HaralickDiffVariance(ans, diff) << endl;
	cout << "Sum Average: " << HaralickSumAverage(ans, sum) << endl;
	cout << "Sum Entropy: " << HaralickSumEntropy(ans, sum) << endl;
	cout << "Sum Variance: " << HaralickSumVariance(ans, sum) << endl;*/


	cout << endl << "Normalize feats: " << endl;
	Mat features = get_global_features(img);
	cout << "global features: " << features.size() << endl;

	//cout << "norm_features features: " <<norm_features.reshape(1,norm_features.cols) << norm_features.size() << endl;
	//cout << "Energy: " << fd_haralik[0] << endl;
	//cout << "Entropy: " << fd_haralik[1] << endl;
	//cout << "Inverse Difference Moment: " << fd_haralik[2] << endl;
	//cout << "Correlation: " << fd_haralik[3] << endl;
	//cout << "Info Measure of Correlation 1: " << fd_haralik[4] << endl;
	//cout << "Info Measure of Correlation 2:" << fd_haralik[5] << endl;
	//cout << "Contrast: " << fd_haralik[6] << endl;
	//cout << "Difference Entropy: " << fd_haralik[7] << endl;
	//cout << "Difference Variance: " << fd_haralik[8] << endl;
	//cout << "Sum Average: " << fd_haralik[9] << endl;
	//cout << "Sum Entropy: " << fd_haralik[10] << endl;
	//cout << "Sum Variance: " << fd_haralik[11] << endl;
	//minMaxLoc(ans, &min, &max);
	//ans = 255 * (ans/max);
	//ans.convertTo(ans, CV_8UC1);
	//imshow("coocslow", ans);
	//waitKey(0);
	//printMat(ans);
	return 0;
}


