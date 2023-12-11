'''python yolov5.py --labelpath coco.names --imgpath sample.jpg --modelpath models/yolov5m.onnx'''

import cv2
import numpy as np
import argparse


# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)


class yolov5():
    def __init__(self, args):
        # define constants.
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.SCORE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.45
        self.CONFIDENCE_THRESHOLD = 0.45
        # define text parameters, adaptable to user preference.
        self.FONT_FACE=cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE=0.7
        self.THICKNESS=1
        self.net = cv2.dnn.readNet(args.modelpath) #get net weights from user input modelpath
        self.classes=None 
        self.dtresults=[]
        self.displayoutput=args.displayoutput

        if (args.disable_cuda==False):
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        #get classes from user input label path
        with open(args.labelpath, 'rt') as f:
             self.classes = f.read().rstrip('\n').split('\n')
    
    def draw_label(self, input_image, label, left, top):
        """Draw text onto image at location."""
        # Get text size.
        text_size = cv2.getTextSize(label, self.FONT_FACE, self.FONT_SCALE, self.THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a BLACK rectangle. 
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
        # Display text inside the rectangle.
        cv2.putText(input_image, label, (left, top + dim[1]), self.FONT_FACE, self.FONT_SCALE, YELLOW, self.THICKNESS, cv2.LINE_AA)
    
    def pre_process(self, input_image):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (self.INPUT_WIDTH, self.INPUT_HEIGHT), [0,0,0], 1, crop=False)
        
        # Sets the input to the network.
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers.
        output_layers = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(output_layers)
        # print(outputs[0].shape)
        return outputs
    

    def post_process(self, input_image, outputs):
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []

        # Rows.
        rows = outputs[0].shape[1]

        image_height, image_width = input_image.shape[:2]

        # Resizing factor.
        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT

        # Iterate through 25200 detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]

            # Discard bad detections and continue.
            if confidence >= self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]

                # Get the index of max class score.
                class_id = np.argmax(classes_scores)

                #  Continue if the class score is above threshold.
                if (classes_scores[class_id] > self.SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)

                    cx, cy, w, h = row[0], row[1], row[2], row[3]

                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
              
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3* self.THICKNESS)
            label = "{}:{:.2f}".format(self.classes[class_ids[i]], confidences[i])
            if (self.displayoutput==True):
                self.draw_label(input_image, label, left, top)
            currentlabel=[class_ids[i], confidences[i], left, top, width, height]
            self.dtresults.append(currentlabel)
        return input_image

    def detect(self,img):
        """function to perform object detection using user defined image"""
        outputs=self.pre_process(img)
        outputimg=self.post_process(img,outputs)
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        print(label)
        if (self.displayoutput==True):
            cv2.putText(outputimg, label, (20, 40), self.FONT_FACE, self.FONT_SCALE, RED, self.THICKNESS, cv2.LINE_AA)
            cv2.imshow('Output', outputimg)
            cv2.waitKey(0)
        
if __name__ == '__main__':
    # Get user input from the command line for the following parameters
    parser = argparse.ArgumentParser(description='object detection')
    parser.add_argument('--disable_cuda', default=True, action='store_true', help='Disable CUDA')
    parser.add_argument('--labelpath', type=str, default='coco.names',help='labels')
    parser.add_argument('--imgpath', type=str, default='sample.jpg', help='imagepath for test')       
    parser.add_argument('--modelpath', type=str, default='model/yolov5m.onnx')
    parser.add_argument('--displayoutput', type=bool, default=True)
    args = parser.parse_args()

    # Load image.
    frame = cv2.imread(args.imgpath)
    #define and initiate model 
    yolonet = yolov5(args)
    # Process image and perform object detection
    yolonet.detect(frame)
    
    

