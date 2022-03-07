import numpy as np
from skimage import exposure


def hist_equalization(image, out_max=None):
    """
    Histogram equalization.
    
    """
    if out_max is None:
        out_max = np.iinfo(image.dtype).max
    return exposure.equalize_hist(image)*out_max


def gamma_correction(image, out_max=None):
    if out_max is None:
        out_max = np.iinfo(image.dtype).max
    return exposure.adjust_gamma(image, 2)/out_max


def windowing(image, cutoff_ratio = (0.0,0.0)):
    """
    image: [h, w]
    cutoff_ratio: [cutoff_left, cutoff_right]
    Returns: rescaled image
    """
    
    orishape = image.shape
    counts_total = orishape[1]*orishape[0] # w*h
    cutoff_left = int(counts_total*cutoff_ratio[0])
    cutoff_right = int(counts_total*(1-cutoff_ratio[1]))
    
    image_histo = image.ravel()

    values, counts = np.unique(image_histo, return_counts=True)
    
    counts_cumsum = np.cumsum(counts)
    
    window_min_idx = np.argmin(counts)
    for i, count_cumsum in enumerate (counts_cumsum):
        if count_cumsum > cutoff_left:
            window_min_idx = i
            break
    window_min_val = values[window_min_idx]
        
    window_max_idx = np.argmax(counts_cumsum)
    for i, count_cumsum in enumerate (counts_cumsum[::-1]):
        if count_cumsum < cutoff_right:
            window_max_idx = len(counts_cumsum)-i-1
            break
    window_max_val = values[window_max_idx]
    
    window_width = window_max_val - window_min_val + 1
    window_center = window_min_val + window_width/2
    image = get_LUT_value_wwwl(image, window_width, window_center, 0, 255)
    
    return image


def get_LUT_value_wwwl(data, width, level, out_min, out_max):
    return np.piecewise(data,
                        [data <= (level - 0.5 - (width-1)/2),
                         data > (level - 0.5 + (width-1)/2)],
                        [out_min, out_max, lambda data: ((data - (level - 0.5))/(width-1) + 0.5)*(out_max-out_min)])


def invert_image(data, dtype=None):
    '''data: array
    Returns: rescaled inverted data'''
        
    if dtype:
        high = np.iinfo(dtype).max
        low = np.iinfo(dtype).min
    else:
        high = data.max()
        low = data.min()
        dtype = data.dtype
        
    data = ~data
    cmin = data.min()
    cmax = data.max()
        
    scale = float(high - low) / (cmax - cmin)
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(dtype)    


def crop_image(image, roi_meta, margin=(0,0,0,0)):
    """Crop image with interest roi location.
    image: [h, w, channel]
    roi_meta: [h, w, y1, x1, y2, x2] : at roi
    margin: [y1, x1, y2, x2] : margin at roi
    cropmeta: [h, w, y1, x1, y2, x2] : at image
    Returns : image(cropped), cropmeta
    """
    src_size = image.shape
    roi_size = roi_meta[:2]
        
    top_y, top_x, bottom_y, bottom_x = roi_meta[2:] + margin*np.asarray((-1, -1, 1, 1))
    if(top_x < 0) : top_x = 0
    if(top_y < 0) : top_y = 0
    if(bottom_x >= roi_size[1]) : bottom_x = roi_size[1]
    if(bottom_y >= roi_size[0]) : bottom_y = roi_size[0]
            
    sizeDiffRatio = np.array(np.divide(src_size[:2],roi_size), dtype="float64")
    top_x = int(top_x*sizeDiffRatio[1])
    top_y = int(top_y*sizeDiffRatio[0])
    bottom_x = int(bottom_x*sizeDiffRatio[1])
    bottom_y = int(bottom_y*sizeDiffRatio[0])

    image_cropped = image[top_y:bottom_y, top_x:bottom_x, :]
        
    return image_cropped, np.concatenate((src_size[:2], [top_y, top_x, bottom_y, bottom_x]), axis=0)


def restore_mask_withCropmeta(image, crop_meta=None):
    """Restore image with cropped information.
    It is only necessary to restore the mask, so the cropped non-roi areas are filled with zeros.
    image: [h, w, channel]
    crop_meta : [h, w, y1, x1, y2, x2] (Loc at Original)
    Returns: [h, w, channel]
    """
    if crop_meta is None:
        return image

    restored_image = np.zeros(np.concatenate((crop_meta[:2], image.shape[2]),axis=None),dtype = image.dtype)
    restored_image[crop_meta[2]:crop_meta[4], crop_meta[3]:crop_meta[5],:] = image
    return restored_image


def crop_image_withCropmeta(image, crop_meta=None):
    """Crop image with already cropped information.
    image: [h, w, channel]
    crop_meta : [h, w, y1, x1, y2, x2] (Loc at Original)
    Returns: [h, w, channel]
    """
    if crop_meta is None:
        return image
    
    assert np.all(image.shape[:2] == crop_meta[:2]), "InputImage Size must be same with Original size."

    top_y, top_x, bottom_y, bottom_x = crop_meta[2:]
    image_cropped = image[top_y:bottom_y, top_x:bottom_x, :]  
    
    return image_cropped


def restore_image_withCropmeta(image, crop_meta=None):
    """Restore image with cropped information.
    It is only necessary to restore the mask, so the cropped non-roi areas are filled with zeros.
    image: [h, w, channel]
    crop_meta : [h, w, y1, x1, y2, x2] (Loc at Original)
    Returns: [h, w, channel]
    """
    if crop_meta is None:
        return image

    restored_image = np.zeros(np.concatenate((crop_meta[:2], image.shape[2]),axis=None),dtype = image.dtype)
    restored_image[crop_meta[2]:crop_meta[4], crop_meta[3]:crop_meta[5],:] = image
    return restored_image


