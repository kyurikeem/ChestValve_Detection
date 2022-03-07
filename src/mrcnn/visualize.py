"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
from PIL import Image
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import skimage.color
import cv2

from mrcnn import utils
import Utils_custom as us


############################################################
#  Visualization
############################################################

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    #random.shuffle(colors)
    return colors


def save_result_figures(image, result, class_names, save_filename,
                        truemasks = None, truemasks_class_id = None,
                        captions=None, show_mask=True, show_bbox=True, title="", figsize=(200, 200), colors=None,  
                        figDir = None, fig_option='all', skip_index=[]):
    plt.switch_backend('agg')
    """
    boxes in result: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks in result: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    truemasks: (optional) [height, width, num_instances] ground-truth mask in image
    truemasks_class_id: [num_instances] in image
    captions: (optional) A list of strings to use as captions for each object
    overlalyDir: (optional) directory of image + instance-masks (just overlay)
    figDir: (optional) directory of image + instance-masks + bboxs + captions, ... with colors
    fig_option: 'all', 'gt', 'ai'
    """
    
    _, ax = plt.subplots(1, 1, figsize=figsize)

    fig = plt.figure()
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()

    height, width  = image.shape[:2]
    ax.set_ylim(height + 0, -0)
    ax.set_xlim(-0, width + 0)
    ax.axis('off')
    ax.set_title(title)

    fig.add_axes(ax)

    masked_image = image.astype(np.uint32).copy()
    masked_image = skimage.color.gray2rgb(masked_image)
    
    ########################### 
    ###### Ground-truth #######
    ########################### 
    if fig_option=='ai':
        truemasks = None
    if truemasks is not None: 
        color = (1.0, 1.0, 1.0)
        
        for i in range(truemasks.shape[2]):
            class_id = truemasks_class_id[i]
            label = class_names[class_id]
            
            _, truemask = cv2.threshold(truemasks[:, :, i], 0, 1, cv2.THRESH_BINARY)
            
            '''Bounding box'''
            bbox_gt = utils.extract_bboxes(np.expand_dims(truemask,-1))[0]
            y1 = bbox_gt[0]; x1 = bbox_gt[1]; y2 = bbox_gt[2]; x2 = bbox_gt[3];
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.5,
                                      alpha=0.7, linestyle="dashed",
                                      edgecolor=color, facecolor='none')
                ax.add_patch(p)
            
            '''Caption'''
            if not captions:
                caption = "{}".format(label)
            else:
                caption = caption[i]
            ax.text(x1, y1 -5, caption, color=color, size=9, backgroundcolor='none')
            
            '''Mask & Polygon'''
            if not show_mask:
                continue
            padded_truemask = np.zeros((truemask.shape[0], truemask.shape[1]), dtype=np.uint8)
            contours = find_contours(padded_truemask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, linewidth = 0.5, facecolor='none', edgecolor=color, alpha = 1)
                ax.add_patch(p)
    
    
    ########################### 
    ####### Prediction ########
    ########################### 
    boxes = np.array(result['rois'])
    masks = np.array(result['masks'])
    class_ids = result['class_ids']
    scores = result['scores']
    N = boxes.shape[0]

    '''Generate random colors'''
    colors = colors or random_colors(N)
    if any(fig_option==np.asarray(['all','ai'])):
        for i in range(N):
            if len(skip_index) >= 1:
                if i in skip_index: 
                    continue

            color = colors[i] 
            box = boxes[i]
            mask = masks[:, :, i]
            class_id = class_ids[i]
            score = scores[i]
                
            '''Bounding box'''
            if not np.any(box):
                continue
            y1, x1, y2, x2 = box
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0.5,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')
                ax.add_patch(p)

            '''Caption'''
            if not captions:
                label = class_names[class_id]
                score = scores[i] if scores is not None else None
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1-5, caption, color=color, size=9, backgroundcolor='none')
            
            '''Mask & Polygon'''
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, linewidth = 0.5, facecolor=color, edgecolor='none',alpha = 0.5)
                ax.add_patch(p)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(masked_image.astype(np.uint8))
        
    if figDir is not None:        
        plt.savefig(os.path.join(figDir,save_filename), dpi = 300, bbox_inches='tight', pad_inches=0)
        plt.cla()


