import os
from yolo5class import yolov5
import numpy as np
import argparse
import cv2
import argparse
from src.load_annotation import load_annotations_gt, load_annotations_dt
from src.evaluators.coco_evaluator import get_coco_summary


def writetofile(detection_results, savedir, filepath):  
    # Define the file name and path
    file_path = savedir + '/' + filepath+'.txt'
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Iterate through detection results and write them to the file
        for result in detection_results:
            # Format the result and write to the file: classid, confidence, left, top, width, height
            file.write(f"{result[0]} {result[1]} {result[2]} {result[3]} {result[4]} {result[5]}\n")

def runmodelonfolder(args):
    '''
    implement python opencv to process first 20 images in folder 
    and save results as yolo format in folder name same as the model
    '''
    yolonet = yolov5(args)
    
    #create folder to save the results
    savedir = args.dir_dets
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        
    #create a list to save inference time of each image in the folder
    inferenceTime=[]  
    
    #perform object detection to the images in the folder
    for filename in os.listdir(args.imgfolder)[:20]:
        imgid=filename.split('.jpg')[0]   
        
        frame = cv2.imread(args.imgfolder+'/'+filename)  
        yolonet.detect(frame)     #perform detection to the current image 
        writetofile(yolonet.dtresults,savedir,imgid)    #save results to a txt file as yolo format
        
        #calculate inference time and save to the list
        t, _ = yolonet.net.getPerfProfile()
        inferenceTime.append(t * 1000.0 / cv2.getTickFrequency())
        
    aveinferTime=np.average(np.array(inferenceTime))  #calculate average inference time
    return aveinferTime

def modelperformance(args):
    '''get model performance using designated images and models'''
    args.dir_dets='test/'+args.modelpath[7:-5]
    
    #run object detection on folder and have the results saved in detection dir
    inferenceTime=runmodelonfolder(args)
    
    #get detection results of all images in coco format
    det_annotations = load_annotations_dt(args.dir_dets,args.imgfolder,args.labelpath)
    if det_annotations is None or len(det_annotations) == 0:
        print(
                'No detection of the selected type was found in the folder.\nCheck if the selected type corresponds to the files in the folder and try again.',
                'Invalid detections')
        
    #get ground truth in coco format
    gt_annotations =load_annotations_gt(args.dir_annotations_gt,args.imgfolder,args.labelpath)      
    if gt_annotations is None or len(gt_annotations) == 0:
        print(
                'No ground-truth bounding box of the selected type was found in the folder.\nCheck if the selected type corresponds to the files in the folder and try again.',
                'Invalid groundtruths')
        
    #to save model evaluation results 
    coco_res = {}
    #pascal_res = {}
    #get evaluation results of model performance and return 
    coco_res = get_coco_summary(gt_annotations, det_annotations)
    coco_res['average inference time']=inferenceTime
    return coco_res

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='object detection')
    #parser.add_argument('--disable_cuda', default=False, action='store_true', help='Disable CUDA')
    parser.add_argument('--labelpath', type=str, default='coco.names',help='labels')
    parser.add_argument('--imgpath', type=str, default='sample.jpg', help='imagepath for test')       
    parser.add_argument('--modelpath', type=str, default='models/yolov5s.onnx')
    parser.add_argument('--displayoutput', type=bool, default=False)
    parser.add_argument('--imgfolder', type=str, default='test/val2017',help='image foler path')
    parser.add_argument('--dir_annotations_gt', type=str,default='test/labels/', help='directory of ground truth, yolo type')
    parser.add_argument('--dir_dets', type=str, default='test/detection/')
    
    
    args = parser.parse_args()
    
    #evaluate yolov5m model using coco dataset and write results to yolov5mresults.txt
    args.modelpath='models/yolov5m.onnx'
    met = modelperformance(args)
    with open('yolov5mresults.txt', 'w') as txtfile:
        for key, value in met.items():
            txtfile.write(f"{key}: {value}\n")
        
    #evaluate yolov5n model using coco dataset and write results to yolov5nresults.txt
    args.modelpath='models/yolov5n.onnx'
    met2 = modelperformance(args)
    with open('yolov5nresults.txt', 'w') as txtfile:
        for key, value in met2.items():
            txtfile.write(f"{key}: {value}\n")

    
    
    
   
