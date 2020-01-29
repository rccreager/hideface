import dlib
#import sys
#import re
#import random
#import os
#from skimage import io 
#from matplotlib import pyplot as plt 
from hideface import tools, utils

img_input_dir = "data/WIDER_train/images/51--Dresses/"
truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
recognizer_dict = {'hog':dlib.get_frontal_face_detector()} 
num_imgs = 10
results = utils.get_labels(img_input_dir, truth_file, recognizer_dict,num_imgs)
print(results[0])
for result in results:
    utils.create_labeled_image(result, 'data/boxed_output')

#for recognizer_name, recognizer in recognizer_dict.items():
#    iou = []
#    for result in results:    
#        best_matches = tools.get_matches_to_truth(result.true_box_list, result.found_box_dict[recognizer_name])
#        for match in best_matches:
#            iou.append(match.true_box.iou(match.found_box))
#        image = io.imread(result.wf_img_name)
#        tools.draw_boxes(image,result.true_box_list, result.found_box_dict[recognizer_name]) 
#        file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", result.wf_img_name)[0][:-4]
#        io.imsave('data/boxed_output/boxed_'+recognizer_name+'_'+file_num+'.jpg', image)


#for label in my_labels:
#    best_matches = tools.get_matches_to_truth(label['true_box_list'], label['recognizer_boxes_list'])
#    for match in best_matches:
#        iou.append(match.true_box.iou(match.found_box))
#    image = io.imread(label['img_name'])
#    tools.draw_boxes(image, label['true_box_list'], label['recognizer_boxes_list'])
#    file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", label['img_name'])[0][:-4]
#    io.imsave('data/boxed_output/boxed_'+file_num+'.jpg', image)

#for file_name in files:
#    
#    file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", file_name)[0][:-4]
#    print('Processing image: ' + str(file_num))
#    image = io.imread(os.path.join(file_path, file_name))
#    face_detector = dlib.get_frontal_face_detector()
#
#    true_box_list = tools.get_ground_truth_boxes(file_name, 'data/wider_face_split/wider_face_train_bbx_gt.txt')
#    if (len(true_box_list) != 1): continue
#    if (true_box_list[0].quality != tools.TruthBoxQuality(0,0,0,0,0,0)): continue
#    
#    found_box_list = tools.get_found_boxes(image, face_detector)
#    best_matches = tools.get_matches_to_truth(true_box_list, found_box_list)
#    for match in best_matches:
#        iou.append(match.true_box.iou(match.found_box))
#    tools.draw_boxes(image, true_box_list, found_box_list)
#    if not (os.path.isdir('data/boxed_output/')): os.mkdir('data/boxed_output/')
#    io.imsave('data/boxed_output/boxed_'+file_num+'.jpg', image)

#plt.hist(iou, bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) 
#plt.title("IoU Histogram") 
#if not (os.path.isdir('data/histos/')): os.mkdir('data/histos/')
#plt.savefig("data/histos/iou.png")    
