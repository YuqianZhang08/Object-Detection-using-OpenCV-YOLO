
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.utils.enumerators import BBFormat, BBType, CoordinatesType

def load_annotations_dt(dir_dets,imgdir,labelpath):        
    '''load yolo format detection annotation from directory to coco format'''                            
    ret = []
    ret = converter.text2bb(dir_dets,bb_type=BBType.DETECTED,
                                    bb_format=BBFormat.XYWH,
                                    type_coordinates=CoordinatesType.ABSOLUTE,
                                    img_dir=imgdir)
    if len(ret) == 0:
            print(
                'No file was found for the selected detection format in the annotations directory.',
                'No file was found')               
    ret = general_utils.replace_id_with_classes(ret, labelpath)           
    return ret                                                                                                                
                                  
def load_annotations_gt(dir_annotations_gt,imgdir,labelpath):
    '''load yolo format ground truth annotation from directory to coco format'''   
    ret = []
    ret = converter.yolo2bb(dir_annotations_gt,
                                    imgdir,
                                    labelpath,
                                    bb_type=BBType.GROUND_TRUTH)
    # Make all types as GT
    [bb.set_bb_type(BBType.GROUND_TRUTH) for bb in ret]
    return ret