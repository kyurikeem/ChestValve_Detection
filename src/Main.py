from DataSet_custom import ModelConfig
from mrcnn import model as modellib
from mrcnn import visualize
from Classifier import DCNN
from Rule_base import *

import os
import numpy as np
import gc
import cv2
from keras import backend as K
from skimage import exposure
import time

def Test_Dataset(savePath, modelConfig, modelPath, testData, maskFp, clsmodelPath = None, 
                 saveFig = False, Cls_thr= 0):
    
    MODEL_DIR = ""
    if os.path.isfile(modelPath):
        MODEL_DIR = os.path.dirname(modelPath)
    elif os.path.isdir(modelPath):
        MODEL_DIR = modelPath
    else:
        print ("No exist path : ",modelPath)
    if(not os.path.isdir(savePath)):
        os.makedirs(savePath)
    
    Figure_DIR_ai, Figure_DIR_gt = None, None
    if saveFig:
        Figure_DIR_ai = savePath + "/Pred" 
        if(not os.path.isdir(Figure_DIR_ai)):
            os.makedirs(Figure_DIR_ai)
        Figure_DIR_gt = savePath + "/GT"
        if(not os.path.isdir(Figure_DIR_gt)):
            os.makedirs(Figure_DIR_gt)
    FPfilter_mask = None
    if maskFp:
        filter_result = []
        FPfilter_mask = cv2.imread(maskFp, 0)
        _, FPfilter_mask = cv2.threshold(FPfilter_mask, 0, 1, cv2.THRESH_BINARY)

    testData.prepare()

    print("Load Model...")
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=modelConfig)
    if os.path.isdir(modelPath):
        modelPath = model.find_last()
    model.load_weights(modelPath, by_name=True)
    if clsmodelPath:
        Re_identity_model = DCNN()
        Re_identity_model.load_weights(clsmodelPath)
    print("Load Model Done !")
    
    Classes = [None]*len(testData.class_info)
    for class_info in testData.class_info:
        Classes[class_info['id']] = class_info['name']
        
    ##############################################################
    # Run detection
    print("Test Run...")
    for i, image_id in enumerate(testData._image_ids) :
        skip_index = []
        testImage, testFileName = testData.load_image_custom(image_id, test=True)
        gt_masks, gt_class_ids = testData.load_mask_custom(image_id, testImage.shape)
        print (i+1," / ",len(testData._image_ids), " : ",testFileName)
        ######################################################
        # Detection
        result = model.detect([testImage], verbose=0)[0] 
        ######################################################
        # ROI True Positive mask(ERegion)
        if maskFp:
            True_count = 0
            masks = result['masks']
            for i_pred in range(len(result['rois'])):
                cur_shape = masks[:, :, i_pred].shape
                mask = masks[:, :, i_pred].astype('uint8')

                copy_fpfilter = cv2.resize(FPfilter_mask, (cur_shape[1], cur_shape[0]), interpolation=cv2.INTER_LINEAR)
                fp_mask = cv2.bitwise_and(copy_fpfilter, mask)

                if (np.sum(fp_mask.astype('uint8')) / np.sum(mask.astype('uint8'))) > 0.05:
                    result['re_identities'].append(True)
                    True_count += 1
                else:
                    filter_result.append(testFileName)
                    result['re_identities'].append(False)

            result = find_FPcls_falseremove(result, True_count)
        ######################################################
        # Classification(Re-identification)
        if clsmodelPath and len(result['rois']) > 0:
            True_count = 0
            
            for u in range(len(result['rois'])):
                cur_roi = result['rois'][u]
                crop_img = testImage[cur_roi[0]:cur_roi[2],cur_roi[1]:cur_roi[3]]
                crop_img = crop_img.astype('uint8')
                crop_img = cv2.resize(crop_img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
                img = exposure.equalize_hist(crop_img)

                cls_result = Re_identity_model.predict(np.expand_dims(img, axis=0))[0][0]
                
                if cls_result > Cls_thr:
                    # result['re_identities'][u] = False
                    result['re_identities'].append(False)
                else:
                    # result['re_identities'][u] = True
                    result['re_identities'].append(True)
                    True_count += 1

            cls_result = find_FPcls_falseremove(result, True_count) 
        else:
            cls_result = result
        ######################################################
        for i in range(10 + 1):
            class_list = find_classindex(cls_result['class_ids'], i)
            if len(class_list) >= 2:
                find_maxscore_afterremove(cls_result, class_list, skip_index)

        # Display & Save 
        saveFileName = os.path.splitext(testFileName)[0] + ".png"
        if saveFig: 
            testImage = testImage[:,:,2]
            visualize.save_result_figures(testImage, cls_result, Classes, saveFileName, \
                                          truemasks = gt_masks, truemasks_class_id = gt_class_ids,\
                                          figDir = Figure_DIR_ai, fig_option='all', skip_index=skip_index)
            visualize.save_result_figures(testImage, cls_result, Classes, saveFileName, \
                                          truemasks = gt_masks, truemasks_class_id = gt_class_ids,\
                                          figDir = Figure_DIR_gt, fig_option='gt', skip_index=skip_index)

    del model
    gc.collect()
    K.clear_session()
    
