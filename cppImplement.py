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
from src.load_annotation import load_annotations_dt, load_annotations_gt
from mdlEvaluation import modelperformance

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
    parser.add_argument('--labelpath', type=str, default='./coco.names',help='labels')
    parser.add_argument('--dllpath', type=str, default='E:/medtronic/git/open/lib/YOLO.dll', help='imagepath for test')       
    parser.add_argument('--modelpath', type=str, default='./model/yolov5s.onnx')
    parser.add_argument('--displayoutput', type=bool, default=False)
    parser.add_argument('--imgfolder', type=str, default='E:/medtronic/Project2/test/val2017',help='image foler path')
    parser.add_argument('--dir_annotations_gt', type=str,default='E:/medtronic/Project2/test/labels', help='directory of ground truth, yolo type')
    parser.add_argument('--dir_dets', type=str, default='./detect')
    parser.add_argument('--pyorcpp', type=str, default='cpp')
    args = parser.parse_args()
    
    #args.modelpath= 'E:/medtronic/git/open/model/yolov5s.onnx'#define the model to use
    
    #Load dll to python. use exact full path
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    lib=kernel32.LoadLibraryW(args.dllpath)
    lib=ctypes.CDLL(args.dllpath)
    
    #define cpp input directory
    class_path = ctypes.c_char_p(args.labelpath.encode('utf-8'))
    model_path = ctypes.c_char_p(args.modelpath.encode('utf-8'))
    resultPath = ctypes.c_char_p(args.dir_dets.encode('utf-8'))
    folderpath = ctypes.c_char_p(args.imgfolder.encode('utf-8'))
    
    '''for single image processing using dll
    
    yolo = YOLO(class_path, model_path) 
    result=yolo.YOLOSingledetect(img_path,imgshow=True)  
    
    return is as defined in ResultOutput'''
    
    if (args.pyorcpp=="cpp"):
        #obtain cpp results runing on a folder of images, with yoyo format results saved in folder
        cppdetect(model_path, class_path, folderpath,resultPath)  
        cppdet_annotations = load_annotations_dt(args.dir_dets,args.imgfolder,args.labelpath)
    
    #load ground truth labels
    gt_annotations =load_annotations_gt(args.dir_annotations_gt,args.imgfolder,args.labelpath)  
    
    #obtain model evaluation results and write to file
    #get evaluation results of model performance and return 
    if (args.pyorcpp=="cpp"):
        cpp_res = {}
        cpp_res = get_coco_summary(gt_annotations, cppdet_annotations)
        with open('E:/medtronic/git/open/cpp5s_res.txt', 'w') as txtfile:
            for key, value in cpp_res.items():
                txtfile.write(f"{key}: {value}\n")
    
    if (args.pyorcpp=="py"):
        #obtain python implementation evaluation results and write to file        
        python_met = modelperformance(args)
        with open('E:/medtronic/git/open/py5s_res.txt', 'w') as txtfile:
            for key, value in python_met.items():
                txtfile.write(f"{key}: {value}\n")