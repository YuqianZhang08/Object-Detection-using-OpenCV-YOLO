import argparse
import cv2
from ctypes import *
import argparse
from test import modelperformance




if __name__ == '__main__':
    # Get user input from the command line for the following parameters
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
    args.modelpath='models/yolov5s.onnx'
    met, inferencetime = modelperformance(args)
    
    
