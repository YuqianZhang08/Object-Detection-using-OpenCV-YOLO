#pragma once
#include "yolov5class.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
using namespace cv::dnn;



Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);



YOLO::YOLO(const string& classpath, const string& mdpath)
{
    modelpath = mdpath;
    ifstream ifs(classpath);
    string line;

    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
	cout << "create YOLO object" << endl;
}

// Draw the predicted bounding box.
void YOLO::draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);

	
}

vector<Mat> YOLO::pre_process(Mat &input_image)
{
    // Convert to blob.
    Net net=readNet(modelpath); 
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    vector<double> layersTimes;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    layerTimes=net.getPerfProfile(layersTimes);
	cout << "preprocess finished" << endl;
    return outputs;
}

ResultOutput YOLO::post_process(Mat input_image, vector<Mat> &outputs)
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes; 
	ResultOutput bbsoutputs;
	SingleResult singlebb;
    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) 
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) 
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) 
    {
        int idx = indices[i];
        Rect box = boxes[idx];

		singlebb.left = box.x;
		singlebb.top = box.y;
		singlebb.width = box.width;
		singlebb.height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(singlebb.left, singlebb.top), Point(singlebb.left + singlebb.width, singlebb.top + singlebb.height), BLUE, 3*THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_list[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, singlebb.left, singlebb.top);
		singlebb.confidence = confidences[idx];
		singlebb.classid = class_ids[idx];
		bbsoutputs.bbresults.push_back(singlebb);
    }

    return bbsoutputs;
}

ResultOutput YOLO::detect(const string& imgpath, bool showimg=true)
{   
	Mat frame;
	frame = imread(imgpath);
    vector<Mat> detections = pre_process(frame);
	result_img = frame.clone();
    ResultOutput imgOutput = post_process(result_img, detections);
    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	double freq = getTickFrequency() / 1000;
	imgOutput.responsetime = layerTimes / freq;

	if (showimg == true) {
		string label = format("Inference time : %.2f ms", imgOutput.responsetime);
		putText(result_img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
		imshow("Output", result_img);
		waitKey(0);
	}
    
    return imgOutput;
}


int main()
{
	// Load class list.
	Mat frame;

	YOLO yolomodel("./coco.names", "./models/yolov5s.onnx");
	ResultOutput output=yolomodel.detect("sample.jpg");
    
	return 0;
}


#ifdef __cplusplus
extern "C" {
#endif

    __declspec(dllexport) YOLO* Yolo_new(const char*  classpath, const char*  modelpath) 
    {
        return new YOLO(classpath, modelpath);
    }

    __declspec(dllexport) ResultOutput Yolo_detect(YOLO* yolo, const char*  imgpath, bool showimg=true)
    {
        return yolo->detect(imgpath,showimg);
    }

    
#ifdef __cplusplus    
}
#endif 