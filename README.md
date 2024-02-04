# Improvement and implementation of C++ and Python code for YOLO object detection with NMS. 

The project is based on the [Object Detection using YOLOv5 (C++/Python)](https://github.com/spmallick/learnopencv/tree/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python). 

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
     >mdlEvaluation.py
     
     Contains code to perform folder image object detection using both yolov5m and yolov5n models with detection results saved at designated folders. The results were then evaluated using coco evaluation methods (average precisions and other metrics), the average inference time is also included. Src include source files for the evaluation.
  
     >cppImplement.py
     
     Contains code to implement cpp opencv dll function to conduct single image object detection and folder image object detection tasks. With results in YOLO format saved in designated directories. Python implementation results will also be generated for comparison. 
  
## Getting Started
Here are the steps on how to execute the code for opencv YOLO object detection. 
## Install Dependencies
```
pip install -r requirements.txt
```
List of tutorials for installing OpenCV for C++ [here](https://learnopencv.com/category/install/).
My environment: Windows 11, Visual Studio C++ 2017, OpenCV 4.8.0 
## Execution
### 1. Python for single image object detection and image results display
```Python
python yolov5class.py
# or use the code below to designate model and images. 
python .\yolo5class.py --imgpath "sample.jpg" --labelpath "coco.names" --modelpath "model/yolov5n.onnx" 
```
### 2. Python for model evaluation using a folder of images
```Python
# indicate the image folder for process, model to use, and also the directory to save individual detection results. Other inputs include --dir_annotations_gt # for the ground truth label path, and --labelpath for class anotations
python mdlEvaluation.py --imgfolder "/cocooimgs" --modelpath "model/yolov5n.onnx" --dir_dets "/labels_dt"
```
Using this code, a txt file will be generated for each image, including the detected object class, confidence, and position. And a final txt file will be generated to show the model performance and inference time

**Compare yolo5n and yolo5m with the same dataset**
'''Python
python mdlEvaluation.py --imgfolder "/cocooimgs" --modelpath "model/yolov5n.onnx" --dir_dets "/labels_dt"
python mdlEvaluation.py --imgfolder "/cocooimgs" --modelpath "model/yolov5m.onnx" --dir_dets "/labels_dt"
'''
Then you will see two files yolov5mresults.txt and yolov5nresults.txt generated in the project folder for comparisons. 

### C++ Windows for single image detection
You can either build the cpp file in visual studio or using CMake, make sure to select the compatible compiler to generate dll for python implementation. 

I used Visual Studio 2017 Win 64 compiler 

Add all of the four cpp project files to source file in CMakeLists. 

```C++ Windows
cmake -S . -B build
cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\main.exe   //for .exe 
```
With the generated exe you will see the object detection result of the sample image

### C++ Windows dll generation for python implementation 
Make change to the CMakeLists.txt 
Using 
>_add_library(YOLO SHARED ${SOURCE_FILES})_

And repeat the cmake process to build dll, which is located in /build/Release

### Using Python for dll implementation
```Python
# to implement cpp YOLO model for object detection through dll
python .\cppImplement.py --dllpath "./lib/YOLO.dll" --imgfolder './test/val2017' --pyorcpp "cpp"

# to implement python YOLO
python .\cppImplement.py --dllpath "./lib/YOLO.dll" --imgfolder './test/val2017' --pyorcpp "py"
```
which generates two .txt files that include the results of cpp implemented model and python implemented model for the task.

## Other functions
In batchprocess.cpp file, the method to implement dll in cpp is also included.


