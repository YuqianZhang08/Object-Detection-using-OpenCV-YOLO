// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
//#include "yolov5.h"
// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;
namespace fs = filesystem;

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);

struct ReturnValue {
    int left;
    int top;
    int height;
    int width;
    int confidence;
    Mat img;
};
// Draw the predicted bounding box.
void draw_label(Mat& input_image, string label, int left, int top)
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


vector<Mat> pre_process(Mat &input_image, Net &net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


ReturnValue post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name) 
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes; 
    ReturnValue result;

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
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
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

        result.left = box.x;
        result.top = box.y;
        result.width = box.width;
        result.height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(result.left, result.top), Point(result.left + result.width, result.top + result.height), BLUE, 3*THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        result.confidence = confidences[idx];
        // Draw class labels.
        draw_label(input_image, label, result.left, result.top);
    }
    result.img=input_image;

    return result;
}

void performfolder(string folderPath)
{
    Mat frame;
    Net net;
    net = readNet("models/yolov5s.onnx"); 
    vector<string> prftimelist;
    vector<ReturnValue> folderResult;

    try {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                std::cout << "Found JPEG file: " << entry.path() << std::endl;
                ofstream outputFile(entry.path());
                frame = imread(entry.path());
                vector<Mat> detections;
                detections = pre_process(frame, net);
                ReturnValue imgresult = post_process(frame.clone(), detections, class_list);
                vector<double> layersTimes;
                double freq = getTickFrequency() / 1000;
                double t = net.getPerfProfile(layersTimes) / freq;
                prftimelist.push_back(t);
                folderResult.push_back(imgresult);
                outputFile << imgresult.left << endl;
                outputFile << imgresult.top << endl;
                outputFile << imgresult.height << endl;
                outputFile << imgresult.width << endl;
                outputFile << imgresult.confidence << endl;
                outputFile.close();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
    }

}

int main()
{
    // Load class list.
    vector<string> class_list;
    ifstream ifs("coco.names");
    string line;

    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    // Load image.
    Mat frame;
    frame = imread("sample.jpg");

    // Load model.
    Net net;
    net = readNet("models/yolov5s.onnx"); 

    vector<Mat> detections;
    detections = pre_process(frame, net);

    ReturnValue img = post_process(frame.clone(), detections, class_list);

    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)

    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time : %.2f ms", t);
    putText(img.img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    imshow("Output", img.img);
    waitKey(0);

    return 0;
}
