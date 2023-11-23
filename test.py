import os
from yolo5class import yolov5
import numpy as np
import argparse
import cv2
from ctypes import *
import argparse
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.utils.enumerators import BBFormat, BBType, CoordinatesType
from src.evaluators.coco_evaluator import get_coco_summary

def load_annotations_dt(args):                                    
    ret = []
    ret = converter.text2bb(args.dir_dets,bb_type=BBType.DETECTED,
                                    bb_format=BBFormat.XYWH,
                                    type_coordinates=CoordinatesType.ABSOLUTE,
                                    img_dir=args.imgdir)
    if len(ret) == 0:
            print(
                'No file was found for the selected detection format in the annotations directory.',
                'No file was found')               
    ret = general_utils.replace_id_with_classes(ret, args.labelpath)           
    return ret                                                                                                                
                                  
def load_annotations_gt(args):
    ret = []
    ret = converter.yolo2bb(args.dir_annotations_gt,
                                    args.imgdir,
                                    args.labelpath,
                                    bb_type=BBType.GROUND_TRUTH)
    # Make all types as GT
    [bb.set_bb_type(BBType.GROUND_TRUTH) for bb in ret]
    return ret

def writetofile(detection_results, savedir, filepath):  
    # Define the file name and path
    file_path = savedir + '/' + filepath+'.txt'
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Iterate through detection results and write them to the file
        for result in detection_results:
            # Format the result and write to the file
            file.write(f"{result[0]} {result[1]} {result[2]} {result[3]} {result[4]} {result[5]}\n")

def runmodelonfolder(args):
    yolonet = yolov5(args)
    savedir = 'test/'+args.modelpath[7:-5]
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    inferenceTime=[]  
    for filename in os.listdir(args.imgdir)[:20]:
        imgid=filename.split('.jpg')[0]   
        frame = cv2.imread(args.imgdir+'/'+filename)
        yolonet.detect(frame)
        writetofile(yolonet.dtresults,savedir,imgid)
        t, _ = yolonet.net.getPerfProfile()
        inferenceTime.append(t * 1000.0 / cv2.getTickFrequency())
    aveinferTime=np.average(np.array(inferenceTime))
    return aveinferTime

def modelperformance(args):
    args.dir_dets='test/'+args.modelpath[7:-5]
    inferenceTime=runmodelonfolder(args)
    det_annotations = load_annotations_dt(args)
    if det_annotations is None or len(det_annotations) == 0:
        print(
                'No detection of the selected type was found in the folder.\nCheck if the selected type corresponds to the files in the folder and try again.',
                'Invalid detections')
    gt_annotations =load_annotations_gt(args)      
    if gt_annotations is None or len(gt_annotations) == 0:
        print(
                'No ground-truth bounding box of the selected type was found in the folder.\nCheck if the selected type corresponds to the files in the folder and try again.',
                'Invalid groundtruths')
    coco_res = {}
    pascal_res = {}

    coco_res = get_coco_summary(gt_annotations, det_annotations)
    return coco_res, inferenceTime

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='object detection')
    #parser.add_argument('--disable_cuda', default=False, action='store_true', help='Disable CUDA')
    parser.add_argument('--labelpath', type=str, default='coco.names',help='labels')
    parser.add_argument('--imgpath', type=str, default='sample.jpg', help='imagepath for test')       
    parser.add_argument('--modelpath', type=str, default='models/yolov5s.onnx')
    parser.add_argument('--displayoutput', type=bool, default=False)
    parser.add_argument('--imgdir', type=str, default='test/val2017',help='image foler path')
    parser.add_argument('--dir_annotations_gt', type=str,default='test/labels/', help='directory of ground truth, yolo type')
    parser.add_argument('--dir_dets', type=str, default='test/detection/')
    
    args = parser.parse_args()
    
    #evaluate yolov5m model using coco dataset
    args.modelpath='models/yolov5m.onnx'
    met, inferencetime = modelperformance(args)
    
    #evaluate yolov5n model using coco dataset
    args.modelpath='models/yolov5n.onnx'
    met2, inferencetime2 = modelperformance(args)

    
    
    
   
