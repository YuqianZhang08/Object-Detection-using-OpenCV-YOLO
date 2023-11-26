#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp> // Include OpenCV headers
#include "yolov5class.h"
#include <filesystem>
#include <windows.h>



void writeResultsToFile(ResultOutput &output, const std::string &filename);
void batchProcess(const std::string& mdlPath, const std::string& classPath, const std::string& folderPath, const std::string& savefolder);
void processViaDll(const char* dllPath, const char* modelPath, const char* classnamePath, const char* imgPath);

