**Improvement of C++ and Python code for YOLO object detection with NMS. **

The project is based on the "Object Detection using YOLOv5 and OpenCV DNN (C++/Python)" from learnopencv repository. 

Major changes and added functions from the original project:
1. Improved the readability, maintenance, and expandability of the code (both yolov5.py, and yolov5.cpp) by grouping the functions, independence, and dependencies into class objects. New files: yolov5class.py and yolov5class.h and yolov5class.cpp. User input from command lines can also be used to define the test pictures.
2. Added two new files test.py, which compares the python and cpp implemented model performance on the same dataset.
compare.py is for the performance analysis for YOLO5n and YOLO5m using the same COCO dataset.

Below is the detailed information. 
