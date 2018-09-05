#include <Windows.h>
#include <iostream>
#include <string>
#include <vector>
#include "feat.h"
#include <opencv2/opencv.hpp>       // opencv general include file
//#include <ml.h>		  // opencv machine learning include file


#define WATER 1
#define FIRE 2;
#define DARK 3;
#define LIGHT 4;
#define WIND 5;


using namespace cv;
using namespace std;
struct TrainSample {
	string file;
	int attr_type;
	Mat features;
} TrainSample_t;
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
			if (hasEnding(tmp,string(".png"))) {
				files.push_back(tmp);
				
			}
	
		} while (FindNextFile(search_handle, &file));

		FindClose(search_handle);
		
		for (std::vector<std::string>::iterator iter = directories.begin(), end = directories.end(); iter != end; ++iter)
			FindFiles(*iter,files);
	}
}
static inline uchar getSampleType(string &fn) {
	if (string::npos != fn.find("fire")) return FIRE;
	if (string::npos != fn.find("wind")) return WIND;
	if (string::npos != fn.find("water")) return WATER;
	if (string::npos != fn.find("dark")) return DARK;
	if (string::npos != fn.find("light")) return LIGHT;
	return 0;
}
static void loadTrainData(vector<string> &files, vector<TrainSample> &dst) {
	for (std::vector<std::string>::iterator fimg = files.begin(), end = files.end(); fimg != end; ++fimg) {
		TrainSample sample;
		sample.file = string(*fimg);
		cout << "loading " << *fimg;
		Mat img = imread(fimg->c_str(), IMREAD_COLOR);
		Mat gf = get_global_features(img);
		sample.features = gf;
		sample.attr_type = getSampleType(*fimg);
		assert(sample.attr_type > 0);
		dst.push_back(sample);
		cout << " . ok " << sample.attr_type <<  endl;
	}
		

}
static void trainClassifier(CvRTrees &rtree,vector<TrainSample> &train_data) {
	int attr_per_sample = train_data[0].features.cols;
	int samples_size = train_data.size();
	Mat training_data = Mat(samples_size, attr_per_sample, CV_32FC1);
	Mat training_classifications = Mat(samples_size, 1, CV_32FC1);
	//load all data into matrices

	for (int i = 0; i != samples_size;i++) {
		Mat f = train_data[i].features;
		f.copyTo(training_data.row(i));
		training_classifications.at<float>(i, 0) = static_cast<float>(train_data[i].attr_type);
	}
	//normalize(training_classifications, training_classifications, 1.0, 0.0, NORM_MINMAX , CV_32FC1);
	// define all the attributes as numerical
	// alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
	// that can be assigned on a per attribute basis

	Mat var_type = Mat(attr_per_sample+1, 1, CV_8U);
	var_type.setTo(Scalar(CV_VAR_NUMERICAL)); // all inputs are numerical

	// this is a classification problem (i.e. predict a discrete number of class
	// outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL
	var_type.at<uchar>(attr_per_sample, 0) = CV_VAR_CATEGORICAL;
	
	//float priors[] = { 1,1,1,1,1};  // weights of each classification for classes
	CvRTParams params = CvRTParams(25, // max depth
		9, // min sample count
		0, // regression accuracy: N/A here
		false, // compute surrogate split, no missing data
		5, // max number of categories (use sub-optimal algorithm for larger numbers)
		nullptr, // the array of priors
		false,  // calculate variable importance
		25,       // number of variables randomly selected at node and used to find the best split(s).
		200,	 // max number of trees in the forest
		0.01f,				// forrest accuracy
		CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination cirteria
	);

	//Taining phase
	cout << "train begin " << endl;
	rtree.train(training_data, CV_ROW_SAMPLE, training_classifications,
		Mat(), Mat(), var_type,Mat(),params);
	cout << "done" << endl;
}

static int assumedType(string fn) {
	if (string::npos != fn.find("test\\f")) return FIRE;
	if (string::npos != fn.find("test\\w")) return WIND;
	if (string::npos != fn.find("test\\a")) return WATER;
	if (string::npos != fn.find("test\\d")) return DARK;
	if (string::npos != fn.find("test\\l")) return LIGHT;
	return -1;
}
int main(int argc, char** argv) {
	if (argc != 2)
	{
		cout << " Usage: display_image ...." << endl;
		return -1;
	}
	string rfcfn = string(argv[1]) + "\\pretrained.xml";
	
	//load all data into matrices
	vector<string> train_files;
	vector<TrainSample> train_data;
	FindFiles(std::string(argv[1])+"\\train", train_files);
	/*for (std::vector<std::string>::iterator iter = train_files.begin(), end = train_files.end(); iter != end; ++iter)
		cout << *iter << endl;*/
	loadTrainData(train_files, train_data);
	CvRTrees classifier;
	trainClassifier(classifier, train_data);
	classifier.save(rfcfn.c_str());
	cout << "saved rfc as " << rfcfn << endl;
	/*for (vector<TrainSample>::iterator iter = train_data.begin(), end = train_data.end(); iter != end; ++iter)
		cout << iter->attr_type <<" "<<iter->file << endl;*/
	vector<string> test_files;
	cout << "train done. testing.." << endl;
	FindFiles(string(argv[1]) + "\\test", test_files);
	vector<int> results;
	for (vector<string>::iterator iter = test_files.begin(), end = test_files.end(); iter != end; ++iter) {
		Mat m = imread(iter->c_str(), IMREAD_COLOR);
		Mat gf = get_global_features(m);
		float pred = classifier.predict(gf, Mat());
		cout << *iter << " prediction: " << pred << endl;
		int correct = (int)pred == assumedType(*iter) ? 1 : 0;
		results.push_back(correct);
	}
	Mat r = Mat(results);
	Scalar r_sum = cv::sum(r);
	double v = r_sum.val[0];
	cout <<"testing over unseen images: " << v << " of " << results.size() << ". error is: " << 100* (results.size()- v)/ results.size()<<"%"<<endl;
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}