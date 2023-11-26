#dll includes the following functions
'''
    __declspec(dllexport) YOLO* Yolo_new(const char*  classpath, const char*  modelpath) 
    {
        return new YOLO(classpath, modelpath);
    }

    __declspec(dllexport) ResultOutput Yolo_detect(YOLO* yolo, const char*  imgpath, bool showimg=true)
    {
        return yolo->detect(imgpath,showimg);
    }
	__declspec(dllexport) void cppbatchProcess(const char* mdlPath, const char* classPath, const char* folderPath, const char* resultPath)
	{
		batchProcess(mdlPath, classPath, folderPath, resultPath);
	}

'''
import ctypes
import os
from ctypes import wintypes
import argparse
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.utils.enumerators import BBFormat, BBType, CoordinatesType
from src.evaluators.coco_evaluator import get_coco_summary
from load_annotation import load_annotations_dt, load_annotations_gt
from modelcompare import modelperformance

#define single output structure type
class SingleResult (ctypes.Structure):
	_fields_= [("classid", ctypes.c_int),
	("confidence", ctypes.c_double),
	("left", ctypes.c_double),
	("top", ctypes.c_double),
	("width", ctypes.c_double),
	("height", ctypes.c_double)]

class ResultOutput (ctypes.Structure):
	_fields_= [("bbresult", SingleResult),
	("responsetime", ctypes.c_double)]
	

#define YOLO object and define input c types 
class YOLO(object):
    def __init__(self, val1,val2):
        lib.Yolo_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.Yolo_new.restype = ctypes.c_void_p
        
        lib.Yolo_detect.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        lib.Yolo_detect.restype =  ResultOutput

        self.obj=lib.Yolo_new(val1,val2)
  
    def YOLOSingledetect(self,imgpath, showIMG):
        return (lib.Yolo_detect(self.obj,imgpath, showIMG))
    
# define function for batch output using loaded cpp function
def cppdetect(mdlpath, classpath, folderpath,resultpath):  
    lib.cppbatchProcess.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]  
    lib.cppbatchProcess(mdlpath, classpath, folderpath,resultpath)


#print(imgOutput)

if __name__ == '__main__':
    # Get user input from the command line for the following parameters
    parser = argparse.ArgumentParser(description='object detection')
    #parser.add_argument('--disable_cuda', default=False, action='store_true', help='Disable CUDA')
    parser.add_argument('--labelpath', type=str, default='coco.names',help='labels')
    parser.add_argument('--imgpath', type=str, default='sample.jpg', help='imagepath for test')       
    parser.add_argument('--modelpath', type=str, default='models/yolov5s.onnx')
    parser.add_argument('--displayoutput', type=bool, default=False)
    parser.add_argument('--imgfolder', type=str, default='test/val2017',help='image foler path')
    parser.add_argument('--dir_annotations_gt', type=str,default='test/labels/', help='directory of ground truth, yolo type')
    parser.add_argument('--dir_dets', type=str, default='test/cppdetection')
    
    args = parser.parse_args()
    
    args.modelpath='models/yolo5s.onnx' #define the model to use
    
    #Load dll to python
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    lib=kernel32.LoadLibraryW('E:/medtronic/medtronic/Project2/x64/Release/Project2.dll')
    os.chdir('E:/medtronic/medtronic/Project2/x64/Release')
    lib=ctypes.CDLL('Project2.dll')
    
    #define cpp input directory
    class_path = ctypes.c_char_p(args.labelpath)
    model_path = ctypes.c_char_p(args.modelpath)
    resultPath = ctypes.c_char_p(args.dir_dets)
    folderpath=ctypes.c_char_p(args.imgfolder)
    
    '''for single image processing using dll
    return is as defined in ResultOutput'''
    #yolo = YOLO(class_path, model_path) 
    #result=yolo.YOLOSingledetect(img_path,imgshow=True)  
    
    #obtain cpp results runing on a folder of images, with yoyo format results saved in folder
    cppdetect(model_path, class_path, folderpath,resultPath)  
    
    #convert yoyo format to coco format for both detection results and ground truth
    cppdet_annotations = load_annotations_dt(args.dir_dets,args.imgfolder,args.labelpath)
    gt_annotations =load_annotations_gt(args.dir_annotations_gt,args.imgfolder,args.labelpath)  
    
    cpp_res = {}
    #get evaluation results of model performance and return 
    cpp_res = get_coco_summary(gt_annotations, cppdet_annotations)
    with open('cpp5s_res.txt', 'w') as txtfile:
        for key, value in cpp_res.items():
            txtfile.write(f"{key}: {value}\n")
            
    python_met = modelperformance(args)
    with open('yolov5sresults.txt', 'w') as txtfile:
        for key, value in python_met.items():
            txtfile.write(f"{key}: {value}\n")