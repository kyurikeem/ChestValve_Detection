import numpy as np

def find_classindex(class_list, val):
    return list(i for i, index in enumerate(class_list) if index == val)


def find_maxscore_afterremove(result, class_list, skip_index):
    maxscore_index = 0
    for i in range(len(class_list)):
        if maxscore_index < result['scores'][class_list[i]]:
            maxscore_index = class_list[i]
    
    class_list.remove(maxscore_index)
    
    skip_index.extend(class_list)


def find_FPcls_falseremove(result, True_count):
    bboxes = result['rois']
    class_ids = result['class_ids']
    masks = result['masks']
    scores = result['scores']
    re_identities = result['re_identities']
    
    count = 0
    shape = masks.shape

    new_result = {'rois':[], 'class_ids':[], 'masks':np.zeros((shape[0],shape[1], True_count), dtype=masks.dtype), 'scores':[], 're_identities':[]}
     
    for i in range(len(re_identities)):
        if re_identities[i] == True:
            new_result['rois'].append(bboxes[i])
            new_result['class_ids'].append(class_ids[i])
            new_result['masks'][:,:,count] = masks[:,:,i]
            new_result['scores'].append(scores[i])
            new_result['re_identities'].append(re_identities[i])

            count += 1
        else:
            pass
        new_result['masks'] = np.array(new_result['masks'])
        
    return new_result