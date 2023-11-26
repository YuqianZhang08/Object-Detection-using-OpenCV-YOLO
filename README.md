# Improvement and implementation of C++ and Python code for YOLO object detection with NMS. 

The project is based on the [Object Detection using YOLOv5 and OpenCV DNN (C++/Python)](https://github.com/spmallick/learnopencv/tree/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python). 

## Features
Major changes and added functions from the original project:
- Created class-based versions of the original code involves structuring the functions into classes, promoting better organization, readability, and maintainability.

  **1. C++ Code Restructuring:**
    >yolov5class.h:
    
    * Creates class declarations, defines member functions, and includes necessary headers.
    * Defines private variables, methods, and class-level properties.
    * Creates new structures to store the object detection boxes, confidence values, and interference time. 
  
    >yolov5class.cpp:
    
     * Implements the functions declared in yolov5class.h.
     * Includes any necessary headers and write the implementation for each member function of the class defined in yolov5class.h.
     * Also defines the functions to be exported in dll file for python implementation.
     * New function “detect” in YOLO class as the only public function to conduct object detection with input image path.
  
  **2. Python code Restructuring:**
    >yolov5class.py:

     * Creates class declaration, defines member functions
     * uses argparse for user-defined model path, image path, and class annotation file.
     * Adds a new function detect with image directory as input for direct object detection execution.
  
- created file **_batchprocess.cpp_** and header file **_batchprocess.h_** to process all images in a designed folder and write YOLO type results to individual txt tiles, which can also be exported as dll library file.
- Included two python files
     >modelcompare.py
     
     Contains code to perform folder image object detection using both yolov5m and yolov5n models with detection results saved at designated folders. The results were then evaluated using coco evaluation methods (average precisions and other metrics), the average inference time is also included.
  
     >cppcompare.py
     
     Contains code to implement cpp opencv dll function to conduct single image object detection and folder image object detection tasks. With results in YOLO format saved in designated directories. Python implementation results will also be generated for comparison. 
  
## Getting Started
