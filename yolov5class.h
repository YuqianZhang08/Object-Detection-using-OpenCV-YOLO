#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
//#include <windows.h>
// A simple class with a constuctor and some methods...
// Namespaces.

#ifndef CONSTANTS_H
#define CONSTANTS_H
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

#endif // CONSTANTS_H


struct SingleResult {
	int classid;
	double confidence;
	double left;
	double top;
	double height;
	double width;
};

struct ResultOutput {
	std::vector<SingleResult> bbresults;
	double responsetime;
};


class YOLO
{
public:
	YOLO(const std::string& classpath, const std::string& modelpath);
	ResultOutput detect(const std::string& imgpath,bool showimg);
private:
	std::vector<std::string> class_list;
	std::string modelpath;
	double layerTimes;
	cv::Mat result_img;
	void draw_label(cv::Mat& input_image, std::string label, int left, int top);
	std::vector<cv::Mat> pre_process(cv::Mat &input_image);
	ResultOutput post_process(cv::Mat input_image, std::vector<cv::Mat> &outputs);
};

