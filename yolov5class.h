#include <opencv2/opencv.hpp>
#include <fstream>

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;



// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);

class YOLO
{
    public:
        YOLO(string classpath, string modelpath);
        void detect(Mat& frame);
    private:
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

        vector<string> class_list;
        Net net;
        void draw_label(Mat& input_image, string label, int left, int top);
        vector<Mat> pre_process(Mat &input_image, Net &net);
        Mat post_process(Mat &input_image, vector<Mat> &outputs); 

};
