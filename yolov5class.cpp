// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>
#include yolov5class.h

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

YOLO::YOLO(string classpath, string modelpath)
{
    this->net = readNet(modelpath);
    ifstream ifs(classpath);
    string line;

    while (getline(ifs, line))
    {
        this->class_list.push_back(line);
    }
}

// Draw the predicted bounding box.
void YOLO::draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, this->FONT_FACE, this->FONT_SCALE, this->THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), this->FONT_FACE, this->FONT_SCALE, YELLOW, this->THICKNESS);
}


vector<Mat> YOLO::pre_process(Mat &input_image, Net &net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(this->INPUT_WIDTH, this->INPUT_HEIGHT), Scalar(), true, false);

    this->net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    this->net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


Mat YOLO::post_process(Mat &input_image, vector<Mat> &outputs) 
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes; 

    // Resizing factor.
    float x_factor = input_image.cols / this->INPUT_WIDTH;
    float y_factor = input_image.rows / this->INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= this->CONFIDENCE_THRESHOLD) 
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, this->class_list.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > this->SCORE_THRESHOLD) 
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
    NMSBoxes(boxes, confidences, this->SCORE_THRESHOLD, this->NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) 
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*this->THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_list[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}

void YOLO::detect(Mat &input_image) 
{
    vector<Mat> detections = this->pre_process(input_image, this->net);
    Mat img = this->post_process(input_image.clone(), detections);
    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)

    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time : %.2f ms", t);
    putText(img, label, Point(20, 40), this->FONT_FACE, this->FONT_SCALE, RED);

    imshow("Output", img);
    waitKey(0);

}

int main()
{
    // Load class list.
    Mat frame;
    frame = imread("sample.jpg");

    YOLO yolomodel("coco.names","models/yolov5s.onnx");
    yolomodel.detect(frame );

    return 0;
}