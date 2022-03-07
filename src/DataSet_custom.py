"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import fnmatch
import numpy as np
import cv2
import csv
import os
import SimpleITK as sitk
from mrcnn import config
from mrcnn import utils
import Utils_custom as us

class ModelConfig(config.Config):
    
    NAME = "xray"  # Override in sub-classes

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200

    BACKBONE = "resnet101"
    RPN_TRAIN_ANCHORS_PER_IMAGE = 32

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
    IMAGE_CHANNEL_COUNT = 3
    TRAIN_ROIS_PER_IMAGE = 50 # 200

    LEARNING_RATE_MODE = "decay"  # constant,decay
    LEARNING_RATE = 0.001
    LEARNING_RATE_DROP = 0.1
    EPOCHS_PER_DROP = 50
    
    LEARNING_MOMENTUM = 0.9

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.
        ,"rpn_bbox_loss": 1.
        ,"mrcnn_class_loss": 1.
        ,"mrcnn_bbox_loss": 1.
        ,"mrcnn_mask_loss": 1.
    }


    def __init__(self, classCount):
        """Set values of computed attributes."""
        self.NUM_CLASSES = 1+classCount
        self.STEPS_PER_EPOCH = self.STEPS_PER_EPOCH / self.IMAGES_PER_GPU
        self.VALIDATION_STEPS = self.VALIDATION_STEPS / self.IMAGES_PER_GPU
        super(ModelConfig, self).__init__()


        
class Dataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    
    def set_dataset(self, imagePath, maskPath, dirDicts, classDicts, roiCSV=None, windowCSV=None):
        """Setting dataset info for load image in realtime.
        dirDicts: key=dirname, val=classname
        classDicts: key=classname, val=classid
        roiCSV: [filename, roi_H, roi_W, y1, x1, y2, x2] in one row
        windowCSV: [filename, window_min1, window_min2, window_max1, window_max2] in one row
        """
        
        if(imagePath is None or not os.path.isdir(imagePath)):
            print ("Wrong ImagePath")
            return
        
        '''Setting class info'''
        for classname in classDicts:
            self.add_class("xray", classDicts[classname], classname)
        
        '''Setting roi meta dict'''
        roiDicts = None
        if roiCSV:
            roiDicts = {}
            f = open(roiCSV, 'r', encoding = "utf-8", newline='')
            f_reader = csv.reader(f)
            for row in f_reader:
                roiDicts.update({row[0]:np.array(row[1:], dtype=np.float16)})
            f.close()
        
        '''Setting window dict'''
        windowDicts = None
        if windowCSV:
            windowDicts = {}
            f = open(windowCSV, 'r', encoding = "utf-8", newline='')
            f_reader = csv.reader(f)
            for row in f_reader:
                windowDicts.update({row[0]:np.array(row[1:], dtype=np.float32)})
            f.close()
        
        '''Arrange dataset for model input'''
        for file in os.listdir(imagePath):
            if os.path.isdir(os.path.join(imagePath,file)):
                continue
            index = len(self.image_info)
            file_id = os.path.splitext(file)[0]
            
            maskFilePaths = []
            maskClasses = []
            if maskPath:
                for dirname in dirDicts:
                    maskPath_cl = os.path.join(maskPath,dirname)
                    for maskFile in os.listdir(maskPath_cl):
                        if fnmatch.fnmatch(maskFile, file_id +'*.png'):
                            maskFilePaths.append(os.path.join(maskPath_cl,maskFile))
                            maskClasses.append(classDicts[dirDicts[dirname]])
            roiMeta = None
            if roiDicts:
                roiMeta = roiDicts[file_id]
            window = None
            if windowDicts:
                window = windowDicts[file_id]
            
            self.add_image("xray", image_id=index, path=os.path.join(imagePath,file), maskPaths=maskFilePaths, maskClasses=maskClasses,\
                           roiMeta=roiMeta, cropMeta=None, window=window)
    
            
    def read_image(self, filePath):
        # Depending on the extension
        # Shape : h, w, channel
        if filePath.endswith(".dcm"):
            image = sitk.ReadImage(filePath)
            pi = image.GetMetaData('0028|0004')
            image = sitk.GetArrayFromImage(image).astype("int16")
            image = np.expand_dims(image[0,:,:], -1)
            if 'MONOCHROME1' in pi:
                image = us.invert_image(image)
        elif filePath.endswith(".png"):
            image = cv2.imread(filePath)
            image = np.array(image, dtype = "uint8")
        elif filePath.endswith(".jpg"):
            image = cv2.imread(filePath)
            image = np.array(image, dtype = "uint8")
        elif filePath.endswith(".mha"):
            image = sitk.ReadImage(filePath)
            image = sitk.GetArrayFromImage(image).astype("int16")
            image = np.transpose(image,(1,2,0))
        else:
            raise ValueError('Data format not supported. '
                'Please check your input data format:', filePath)
        return image
    
    
    def update_info(self, image_id, key, value):
        self.image_info[image_id][key] = value

    
    def load_image_custom(self, image_id, test=False):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        
        info = self.image_info[image_id]
        filePath = info["path"]
        roi_meta = info["roiMeta"]
        window = info["window"]
        
        '''Load image'''
        filename = os.path.basename(filePath)
        data = self.read_image(filePath)[:,:,0]
        '''Cropping'''
        crop_meta=None
        if roi_meta is not None:
            if test:
                crop_margin = np.array((0,0,0,0))
            else:
                crop_margin = np.random.randint(0, 30, 4)
                crop_margin[2] = np.random.randint(0, 150, 1)[0]
            data, crop_meta = us.crop_image(np.expand_dims(data, -1), roi_meta, margin=crop_margin)
            data = data[:,:,0]
            self.update_info(image_id, "cropMeta", crop_meta)
            
        '''Hist Equalization'''
        data_hEq = us.hist_equalization(data,255)
        data_hEq = us.gamma_correction(data_hEq, 255)

        '''Windowing'''
        if window is not None:
            if test:
                window_min = np.mean(window[:2])
                window_max = np.mean(window[2:])
            else:
                window_min = np.random.randint(min(window[:2])-200, max(window[:2])+200, 1)[0]
                window_max = np.random.randint(min(window[2:])-200, max(window[2:])+200, 1)[0]

        image = []
        image.append(data_hEq)
        image.append(data_hEq)
        image.append(data_hEq)

        image = np.transpose(image,(1,2,0))
        image = np.array(image, dtype=data.dtype) # data.dtype
        
        return image, filename
    
    
    def load_mask_custom(self, image_id, image_shape):
        """Generate instance masks for shapes of the given image ID.
        image_shape: [h, w]
        Returns: [h, w, instance_count]
        """
        info = self.image_info[image_id]
        filePaths = info['maskPaths']
        classes = info['maskClasses']
        crop_meta = info['cropMeta']
        if crop_meta is not None:
            image_shape = crop_meta[:2]
        
        masks = []
        class_ids = []
        
        '''Load masks'''
        # 1 filePath -- 1 class
        for i, filePath in enumerate(filePaths):
            
            mask = cv2.imread(filePath, 0)
            mask = np.asarray(mask, dtype = "uint8")
                
            masks.append(mask)
            class_ids.append(classes[i])
            
        if len(masks)==0 :
            masks.append(np.zeros(image_shape, dtype = "uint8"))
            class_ids.append(0)
        
        masks = np.stack(masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        ''''Cropping'''
        masks = us.crop_image_withCropmeta(masks, crop_meta)
        
        return masks, class_ids
    
    
    def get_cropMeta(self, image_id):
        return self.image_info[image_id]['cropMeta']