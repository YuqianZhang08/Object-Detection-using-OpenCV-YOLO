#include "batchprocess.h"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;



void writeResultsToFile(ResultOutput &output, const string &filename) {
	ofstream outFile(filename);

	if (!outFile.is_open()) {
		cerr << "Unable to open file " << filename << endl;
		return;
	}

	// Write bounding box results to the file: classid confidence left top width height
	for (int i = 0; i < output.NumObject; i++)
	{
		SingleResult result = output.bbresults[i];
		outFile << result.classid << " " << result.confidence <<" "<< result.left << " " << result.top
			<< " " << result.width << " " << result.height
			 << endl;
	}

	outFile.close();
}

void batchProcess(const string& mdlPath, const string& classPath, const string& folderPath, const string& savefolder) {

	//create yolomodel and initiate the dnn model with classes
	YOLO yolomodel(classPath, mdlPath);

	//disable image showing for batch processing
	bool showimg = FALSE;
	//generate vector to save individual response time of all images included
	vector<double> responsetime;

	//process images in the designated folder
	try {
		for (const auto& entry : fs::directory_iterator(folderPath)) {
			if (entry.path().extension() == ".jpg") {
				cout << "Found JPEG file: " << entry.path() << endl;
				//obtain indivisual image results (classes, boxes, and confidence)
				ResultOutput result = yolomodel.detect(entry.path().string(), showimg);

				//write results of each images to individual files and save in detect folder
				if (!fs::exists(savefolder))
				{
					fs::create_directory(savefolder);
				}
				//save to txt file with name same as the image's
				string filename = savefolder+ '/' +entry.path().filename().string().substr(0, entry.path().filename().string().size()-4) + ".txt";
				writeResultsToFile(result, filename);

				//add current image process time to vector
				responsetime.push_back(result.responsetime);
			}
		}
	
	}
	catch (const std::exception& e) {
		cerr << "Error accessing directory: " << e.what() << endl;
	}
	//save response time to the results folder
	ofstream outputFile(savefolder+"/responseTime.txt");
	if (!outputFile.is_open()) {
		cerr << "Error opening the file." << endl;
	}

	// Write numbers to the file
	for (const auto& rstime : responsetime) {
		outputFile << rstime << "ms\n"; // Write each number followed by a newline
	}

	// Close the file
	outputFile.close();

}

void processViaDll(const char* dllPath, const char* modelPath, const char* classnamePath, const char* imgPath) {
	//process the images using dll instead of the cpp executive functions 

	typedef YOLO* (*Yolo_new)(const char*, const char*);
	typedef ResultOutput(*Yolo_detect)(YOLO*, const char*);

	HMODULE hDLL = LoadLibrary(dllPath);

	if (hDLL != NULL) {
		// Get function pointers
		Yolo_new yoloNewFunc = reinterpret_cast<Yolo_new>(GetProcAddress(hDLL, "Yolo_new"));
		Yolo_detect yoloDetectFunc = reinterpret_cast<Yolo_detect>(GetProcAddress(hDLL, "Yolo_detect"));

		if (yoloNewFunc != NULL && yoloDetectFunc != NULL) {
			// Use the functions from the DLL
			YOLO* yolo = yoloNewFunc(classnamePath, modelPath);
			ResultOutput result = yoloDetectFunc(yolo, imgPath);
		}
		else {
			std::cerr << "Failed to get function pointers." << std::endl;
		}

		// Free the loaded DLL
		FreeLibrary(hDLL);
	}
	else {
		std::cerr << "Failed to load the DLL." << std::endl;
	}


}


int main() {
	
	string folderPath = "E:/medtronic/Project2/test/val2017"; // Update this with your folder path

	string classname = "E:/medtronic/coco.names"; 

	string modelPath = "E:/medtronic/Project2/models/yolov5s.onnx";
	string savepath = "E:/medtronic/Project2/test/cppdetect";

	//YOLO yolomodel(classname, modelPath);

	batchProcess( modelPath, classname, folderPath, savepath);  //process the images in a folder
	
	return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
	//export batch process function to dll and call with other platforms.

	__declspec(dllexport) void cppbatchProcess(const char* mdlPath, const char* classPath, const char* folderPath, const char* savepath)
	{
		batchProcess(mdlPath, classPath, folderPath,savepath);
	}


#ifdef __cplusplus    
}
#endif 